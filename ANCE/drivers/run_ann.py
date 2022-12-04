import sys
sys.path += ['../']
import os
import torch
from data.msmarco_data import GetTrainingDataProcessingFn, GetTripletTrainingDataProcessingFn
from utils.util import (
    getattr_recursive,
    set_seed,
    StreamingDataset,
    EmbeddingCache,
    get_checkpoint_no,
    get_latest_ann_data,
    get_latest_group_result,
    is_first_worker
)
import pandas as pd
from transformers import glue_processors as processors
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup
)
from model.dro_greedy_loss import AverageMeter
import transformers
from utils.lamb import Lamb
from model.models import MSMarcoConfigDict, ALL_MODELS
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
from os.path import isfile, join
import argparse
import glob
import json
import logging
import random
from utils.eval_mrr import passage_dist_eval_ance

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter
logger = logging.getLogger(__name__)

def get_latest_checkpoint(args):
    if not os.path.exists(args.training_dir):
        return args.init_model_dir, 0
    subdirectories = list(next(os.walk(args.training_dir))[1])
    
    def valid_checkpoint(checkpoint):
        chk_path = os.path.join(args.training_dir, checkpoint)
        scheduler_path = os.path.join(chk_path, "scheduler.pt")
        return os.path.exists(scheduler_path)

    checkpoint_nums = [get_checkpoint_no(
        s) for s in subdirectories if valid_checkpoint(s)]

    if len(checkpoint_nums) > 0:
        return os.path.join(args.training_dir, "checkpoint-" +
                            str(max(checkpoint_nums))) + "/", max(checkpoint_nums)
    return args.init_model_dir, 0


def train(args, model, tokenizer, query_cache, passage_cache):
    """ Train the model """
    logger.info("Training/evaluation parameters %s", args)
    tb_writer = None
    if is_first_worker():
        tb_writer = SummaryWriter(log_dir=args.log_dir)

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    real_batch_size = args.train_batch_size * args.gradient_accumulation_steps * \
        (torch.distributed.get_world_size() if args.local_rank != -1 else 1)

    optimizer_grouped_parameters = []
    layer_optim_params = set()
    for layer_name in [
        "roberta.embeddings",
        "score_out",
        "downsample1",
        "downsample2",
        "downsample3"]:
        layer = getattr_recursive(model, layer_name)
        if layer is not None:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)
    if getattr_recursive(model, "roberta.encoder.layer") is not None:
        for layer in model.roberta.encoder.layer:
            optimizer_grouped_parameters.append({"params": layer.parameters()})
            for p in layer.parameters():
                layer_optim_params.add(p)

    optimizer_grouped_parameters.append(
        {"params": [p for p in model.parameters() if p not in layer_optim_params]})

    global_step = 0
    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        # set global_step to gobal_step of last saved checkpoint from model
        # path
        if "-" in args.model_name_or_path:
            try:
                global_step = int(
                    args.model_name_or_path.split("-")[-1].split("/")[0])
            except ValueError:
                global_step = 0
        else:
            global_step = 0
        logger.info(
            "[ANN]  Continuing training from checkpoint, will skip to saved global_step")
        logger.info("[ANN] Continuing training from global step %d", global_step)
        args.max_steps += global_step
        total_training_steps = int(args.total_training_steps)
        if global_step < total_training_steps:
            decay_lr = (1 - global_step/total_training_steps)
            args.learning_rate = max(0.2, decay_lr) * args.learning_rate
        else:
            args.learning_rate = 0.2 * args.learning_rate

        logger.info("[ANN] Max Steps %d", args.max_steps)
        if is_first_worker():
            print("[ANN] Continuing training from global step %d"%global_step)
            print("Max Steps: %d"%(args.max_steps))
            print(f"Max Learning Rate: {args.learning_rate}")
        

    if args.optimizer.lower() == "lamb":
        optimizer = Lamb(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon)
    elif args.optimizer.lower() == "adamw":
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            eps=args.adam_epsilon)
    else:
        raise Exception(
            "optimizer {0} not recognized! Can only be lamb or adamW".format(
                args.optimizer))

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(
        os.path.join(
            args.model_name_or_path,
            "optimizer.pt")) and args.load_optimizer_scheduler:
        # Load in optimizer and scheduler states
        optimizer.load_state_dict(
            torch.load(
                os.path.join(
                    args.model_name_or_path,
                    "optimizer.pt")))

    if args.fp16:
        try:
            sys.path.append("/data/chenyan/workspace/apex")
            from apex import amp
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(
            model, optimizer, opt_level=args.fp16_opt_level)
    
    
    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train
    logger.info("[ANN]***** Running training *****")
    logger.info("[ANN]  Max steps = %d", args.max_steps)
    logger.info(
        "[ANN]  Instantaneous batch size per GPU = %d",
        args.per_gpu_train_batch_size)
    logger.info(
        "[ANN]  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.train_batch_size
        * args.gradient_accumulation_steps
        * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info(
        "[ANN]  Gradient Accumulation steps = %d",
        args.gradient_accumulation_steps)

    tr_loss = 0.0
    model.zero_grad()
    model.train()
    set_seed(args)  # Added here for reproductibility

    last_ann_no = -1
    train_dataloader = None
    train_dataloader_iter = None
    dev_ndcg = 0
    step = 0

    if args.single_warmup:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=args.warmup_steps,
            num_training_steps=args.num_training_steps)

    # init_decay = 1.0
    while global_step <= args.max_steps:

        if step == 0 or (step % args.gradient_accumulation_steps == 0 and global_step % args.logging_steps == 0):
            # check if new ann training data is availabe
            try:
                ann_no, ann_path, ndcg_json, ndcg_beir = get_latest_ann_data(args.ann_dir, args.result_dir)
                update = 1
                
            except:
                ann_no, ann_path, ndcg_json = get_latest_ann_data(args.ann_dir, args.result_dir)
                update = 0
            if ann_path is not None and ann_no != last_ann_no and update:
                logger.info("[ANN] Training on new add data at %s", ann_path)
                dev_ndcg = ndcg_json['ndcg']
                dev_mrr = ndcg_json['mrr']
                ann_checkpoint_path = ndcg_json['checkpoint']
                ann_checkpoint_no = get_checkpoint_no(ann_checkpoint_path)

                if args.stop_iter < 0 or ann_no < args.stop_iter:
                    # if global_step > 0:
                    with open(ann_path, 'r') as f:
                        ann_training_data = f.readlines()
                    
                    aligned_size = (len(ann_training_data) //
                                    args.world_size) * args.world_size
                    ann_training_data = ann_training_data[:aligned_size]

                    logger.info("[ANN] Total ann queries: %d", len(ann_training_data))
                    if args.triplet:
                        train_dataset = StreamingDataset(
                            ann_training_data, GetTripletTrainingDataProcessingFn(
                                args, query_cache, passage_cache), size = len(query_cache))
                    else:
                        train_dataset = StreamingDataset(
                            ann_training_data, GetTrainingDataProcessingFn(
                                args, query_cache, passage_cache), size = len(query_cache))
                    train_dataloader = DataLoader(
                        train_dataset, batch_size=args.train_batch_size)
                    train_dataloader_iter = iter(train_dataloader)

                    # re-warmup
                    if not args.single_warmup:
                        # if not args.cycle_warmup:
                        scheduler = get_linear_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps = args.warmup_steps,
                            num_training_steps = len(ann_training_data))
                if args.local_rank != -1:
                    dist.barrier()

                if is_first_worker():
                    # add ndcg at checkpoint step used instead of current step
                    tb_writer.add_scalar(
                        "dev_ndcg", dev_ndcg, ann_checkpoint_no)
                    tb_writer.add_scalar(
                        "dev_mrr", dev_mrr, ann_checkpoint_no)

                    for key, value in ndcg_beir.items():
                        tb_writer.add_scalar("test_ndcg_" + key, value, ann_checkpoint_no)
                    if last_ann_no != -1:
                        tb_writer.add_scalar(
                            "epoch", last_ann_no, global_step - 1)
                    tb_writer.add_scalar("epoch", ann_no, global_step)
                if args.local_rank != -1:
                    dist.barrier()
                last_ann_no = ann_no

        try:
            batch = next(train_dataloader_iter)
        except StopIteration:
            logger.info("[ANN] Finished iterating current dataset, begin reiterate")
            train_dataloader_iter = iter(train_dataloader)
            batch = next(train_dataloader_iter)
        model.train()
        batch = tuple(t.to(args.device) for t in batch)
        step += 1

        if args.triplet:
           
            inputs = {
                    "query_ids": batch[0].long(),
                    "attention_mask_q": batch[1].long(),
                    "input_ids_a": batch[3].long(),
                    "attention_mask_a": batch[4].long(),
                    "input_ids_b": batch[6].long(),
                    "attention_mask_b": batch[7].long(),
                    "group_ids": batch[9].long() if args.group == 1 else None,
                    "weights": batch[10].float(),
                }
        else:
            inputs = {
                    "input_ids_a": batch[0].long(),
                    "attention_mask_a": batch[1].long(),
                    "input_ids_b": batch[3].long(),
                    "attention_mask_b": batch[4].long(),
                    "labels": batch[6],
                }

        # sync gradients only at gradient accumulation step
        if step % args.gradient_accumulation_steps == 0:
            outputs = model(**inputs)
        else:
            with model.no_sync():
                outputs = model(**inputs)
        # model outputs are always tuple in transformers (see doc)
        loss = outputs[0]
        
        if args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
           
        if args.fp16:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            if step % args.gradient_accumulation_steps == 0:
                loss.backward()
            else:
                with model.no_sync():
                    loss.backward()

        tr_loss += loss.item()

        if step % args.gradient_accumulation_steps == 0:
            if args.fp16:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                logs = {}
                
                loss_scalar = tr_loss / args.logging_steps
                learning_rate_scalar = scheduler.get_lr()[0]
                logs["learning_rate"] = learning_rate_scalar                
                logs["loss"] = loss_scalar
                logs["grad_norm"] = grad_norm.detach().cpu().item()
                tr_loss = 0

                if is_first_worker():
                    for key, value in logs.items():
                        tb_writer.add_scalar(key, value, global_step)
                    logger.info(json.dumps({**logs, **{"step": global_step}}))
                    
                if args.local_rank != -1:
                    dist.barrier()

            if is_first_worker() and args.save_steps > 0 and global_step % args.save_steps == 0:
                # Save model checkpoint
                output_dir = os.path.join(
                    args.output_dir, "checkpoint-{}".format(global_step))
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                model_to_save = (
                    model.module if hasattr(model, "module") else model
                )  # Take care of distributed/parallel training
                model_to_save.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)

                torch.save(args, os.path.join(output_dir, "training_args.bin"))
                logger.info("Saving model checkpoint to %s", output_dir)

                torch.save(
                    optimizer.state_dict(),
                    os.path.join(
                        output_dir,
                        "optimizer.pt"))
                torch.save(
                    scheduler.state_dict(),
                    os.path.join(
                        output_dir,
                        "scheduler.pt"))
                logger.info(
                    "Saving optimizer and scheduler states to %s",
                    output_dir)
            if args.local_rank != -1:
                dist.barrier()
            

    if args.local_rank == -1 or torch.distributed.get_rank() == 0:
        tb_writer.close()

    return global_step


def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the cached passage and query files",
    )

    parser.add_argument(
        "--training_dir",
        default=None,
        type=str,
        required=True,
        help="Training dir, will look for latest checkpoint dir in here",
    )

    parser.add_argument(
        "--init_model_dir",
        default=None,
        type=str,
        required=True,
        help="The ann training data dir. Should contain the output of ann data generation job",
    )

    parser.add_argument(
        "--ann_dir",
        default=None,
        type=str,
        required=True,
        help="The ann training data dir. Should contain the output of ann data generation job",
    )

    parser.add_argument(
        "--result_dir",
        default=None,
        type=str,
        required=True,
        help="The result file dir. Should contain the output of evaluation results",
    )

    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " +
        ", ".join(
            MSMarcoConfigDict.keys()),
    )

    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: " +
        ", ".join(ALL_MODELS),
    )

    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " +
        ", ".join(
            processors.keys()),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )

    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--max_query_length",
        default=64,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )

    parser.add_argument(
        "--triplet",
        default=False,
        action="store_true",
        help="Whether to run training.",
    )

    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--log_dir",
        default=None,
        type=str,
        help="Tensorboard log dir",
    )

    parser.add_argument(
        "--optimizer",
        default="lamb",
        type=str,
        help="Optimizer - lamb or adamW",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )

    parser.add_argument(
        "--learning_rate",
        default=5e-5,
        type=float,
        help="The initial learning rate for Adam.",
    )

    parser.add_argument(
        "--weight_decay",
        default=0.0,
        type=float,
        help="Weight decay if we apply some.",
    )

    parser.add_argument(
        "--adam_epsilon",
        default=1e-8,
        type=float,
        help="Epsilon for Adam optimizer.",
    )

    parser.add_argument(
        "--max_grad_norm",
        default=1.0,
        type=float,
        help="Max gradient norm.",
    )

    parser.add_argument(
        "--max_steps",
        default=1000000,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )

    parser.add_argument(
        "--stop_iter",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform",
    )

    parser.add_argument(
        "--warmup_steps",
        default=0,
        type=int,
        help="Linear warmup over warmup_steps.",
    )

    parser.add_argument(
        "--num_training_steps",
        default=60000,
        type=int,
        help="Number of the maximum training steps in each episode.",
    )

    parser.add_argument(
        "--total_training_steps",
        default=300000,
        type=int,
        help="Maximum training steps in total.",
    )

    parser.add_argument(
        "--logging_steps",
        type=int,
        default=500,
        help="Log every X updates steps.",
    )

    parser.add_argument(
        "--evaluate_during_training",
        type=int,
        default=1,
        help="Log every X updates steps.",
    )
    
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save checkpoint every X updates steps.",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available",
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for initialization",
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )

    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )

    # ----------------- ANN HyperParam ------------------

    parser.add_argument(
        "--load_optimizer_scheduler",
        default=False,
        action="store_true",
        help="load scheduler from checkpoint or not",
    )

    parser.add_argument(
        "--single_warmup",
        default=False,
        action="store_true",
        help="use single or re-warmup",
    )

    parser.add_argument(
        "--cycle_warmup",
        default=False,
        action="store_true",
        help="use cycle or not",
    )

    # ----------------- End of Doc Ranking HyperParam ------------------
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=-1, 
        help="For distributed training: local_rank",
    )
    
    parser.add_argument(
        "--server_ip",
        type=str,
        default="",
        help="For distant debugging.",
    )
    
    parser.add_argument(
        "--server_port",
        type=str,
        default="",
        help="For distant debugging.",
    )
    
    parser.add_argument(
        "--group",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--n_groups",
        default=0,
        type=int,
    )

    parser.add_argument(
        "--alpha",
        default=0,
        type=float,
        help="'alpha for greedy group DRO",
    )

    parser.add_argument(
        "--beta",
        default=0,
        type=float,
        help="beta for greedy group DRO",
    )

    parser.add_argument(
        "--eps",
        default=0.1,
        type=float,
        help="eps for weight for NOT selected group greedy group DRO",
    )

    parser.add_argument(
        "--rho",
        default=0.1,
        type=float,
        help="eps for weight for NOT selected group greedy group DRO",
    )

    parser.add_argument(
        "--dro_type",
        default='dro-greedy',
        type=str,
        help="ema for weight and loss smoothing for group greedy group DRO",
    )

    parser.add_argument(
        "--ema",
        default=0.1,
        type=float,
        help="ema for weight and loss smoothing for group greedy group DRO",
    )

    parser.add_argument(
        "--weight_ema",
        default=0,
        type=int,
        help="whether use ema for weight smoothing",
    )

    
    parser.add_argument(
        "--round",
        default=0,
        type=int,
        help="round",
    )

    parser.add_argument(
        "--model_size",
        default='large',
        type=str,
        help="Whether base or large.",
    )
    args = parser.parse_args()

    return args


def set_env(args):

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see
        # https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(
                args.server_ip,
                args.server_port),
            redirect_output=True)
        ptvsd.wait_for_attach()

    args.rank =  int(dict(os.environ)["RANK"])
    args.server_ip = dict(os.environ)["MASTER_ADDR"]
    args.server_port = dict(os.environ)["MASTER_PORT"]
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl",rank=args.rank)
        args.n_gpu = 1
    args.device = torch.device("cuda", args.local_rank)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    # Set seed
    set_seed(args)


def load_model(args):
    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    args.output_mode = "classification"
    label_list = ["0", "1"]
    num_labels = len(label_list)

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()
    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()
    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    config = configObj.config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = configObj.model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    if args.group == 1:
        model.add_group_loss(args = args, n_groups = args.n_groups, dro_type= args.dro_type, \
                        alpha = args.alpha, eps = args.eps, ema = args.ema, \
                        rho = args.rho, weight_ema = bool(args.weight_ema))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()
    
    model.to(args.device)

    return tokenizer, model

def load_model_ckpt(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
        # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    args.model_type = args.model_type.lower()
    configObj = MSMarcoConfigDict[args.model_type]
    args.model_name_or_path = checkpoint_path
    config = configObj.config_class.from_pretrained(
        args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task="MSMarco",
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    tokenizer = configObj.tokenizer_class.from_pretrained(
        args.model_name_or_path,
        do_lower_case=True,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    model = configObj.model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    
    if args.group == 1:
        model.add_group_loss(args = args, n_groups = args.n_groups, dro_type= args.dro_type, \
                        alpha = args.alpha, eps = args.eps, ema = args.ema, \
                        rho = args.rho, weight_ema = bool(args.weight_ema))

    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()

    model.to(args.device)
    logger.info("Inference parameters %s", args)

    return config, tokenizer, model

def save_checkpoint(args, model, tokenizer):
    # Saving best-practices: if you use defaults names for the model, you can
    # reload it using from_pretrained()
    if args.local_rank != -1:
        dist.barrier()
    if is_first_worker():
        # Create output directory if needed
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained
        # model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

    if args.local_rank != -1:
        dist.barrier()
    return


def main():
    args = get_arguments()
    set_env(args)
    if args.round == 0:
        tokenizer, model = load_model(args)
    else:
        next_checkpoint, latest_step_num = get_latest_checkpoint(args)
        config, tokenizer, model = load_model_ckpt(args, next_checkpoint)

    query_collection_path = os.path.join(args.data_dir, "train-query")
    query_cache = EmbeddingCache(query_collection_path, group  = bool(args.group))
    passage_collection_path = os.path.join(args.data_dir, "passages")
    passage_cache = EmbeddingCache(passage_collection_path, group = bool(args.group))

    with query_cache, passage_cache:
        global_step = train(args, model, tokenizer, query_cache, passage_cache)
        logger.info(" global_step = %s", global_step)

    save_checkpoint(args, model, tokenizer)
    return 


if __name__ == "__main__":
    main()
