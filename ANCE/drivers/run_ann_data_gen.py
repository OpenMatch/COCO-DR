import sys
sys.path += ['../']
import torch
import os
import faiss
from utils.util import (
    barrier_array_merge,
    convert_to_string_id,
    is_first_worker,
    StreamingDataset,
    EmbeddingCache,
    get_checkpoint_no,
    get_latest_ann_data
)
import csv
import copy
import transformers
from transformers import (
    AdamW,
    RobertaConfig,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
    RobertaModel,
)
from data.msmarco_data import GetProcessingFn  
from model.models import MSMarcoConfigDict, ALL_MODELS
from torch import nn
import torch.distributed as dist
from tqdm import tqdm, trange
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import numpy as np
from os.path import isfile, join
import argparse
import json
import logging
import random
import time
import pytrec_eval
import pickle 
from collections import defaultdict

torch.multiprocessing.set_sharing_strategy('file_system')


logger = logging.getLogger(__name__)


# ANN - active learning ------------------------------------------------------

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


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


def load_positive_ids(args):

    logger.info("Loading query_2_pos_docid")
    training_query_positive_id = {}
    query_positive_id_path = os.path.join(args.data_dir, "train-qrel.tsv")
    try:
        with open(query_positive_id_path, 'r', encoding='utf8') as f:
            tsvreader = csv.reader(f, delimiter="\t")
            for [topicid, docid, rel] in tsvreader:
                assert rel == "1"
                topicid = int(topicid)
                docid = int(docid)
                training_query_positive_id[topicid] = docid
    except FileNotFoundError:
        print('Warning: Traininig positive query id not uploaded!')
        training_query_positive_id = []

    logger.info("Loading dev query_2_pos_docid")
    dev_query_positive_id = {}
    query_positive_id_path = os.path.join(args.data_dir, "dev-qrel.tsv")

    with open(query_positive_id_path, 'r', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter="\t")
        for [topicid, docid, rel] in tsvreader:
            topicid = int(topicid)
            docid = int(docid)
            if topicid not in dev_query_positive_id:
                dev_query_positive_id[topicid] = {}
            dev_query_positive_id[topicid][docid] = int(rel)

    return training_query_positive_id, dev_query_positive_id

def load_model(args, checkpoint_path):
    label_list = ["0", "1"]
    num_labels = len(label_list)
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
    if args.local_rank == 0:
        # Make sure only the first process in distributed training will
        # download model & vocab
        torch.distributed.barrier()
    model.to(args.device)
    logger.info("Inference parameters %s", args)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[
                args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )
    return config, tokenizer, model


def InferenceEmbeddingFromStreamDataLoader(
        args,
        model,
        train_dataloader,
        is_query_inference=True,
        prefix=""):
    # expect dataset from ReconstructTrainingSet
    results = {}
    eval_batch_size = args.per_gpu_eval_batch_size

    # Inference!
    logger.info("***** Running ANN Embedding Inference *****")
    logger.info("  Batch size = %d", eval_batch_size)

    embedding = []
    embedding2id = []

    if args.local_rank != -1:
        dist.barrier()
    model.eval()

    for batch in tqdm(train_dataloader,
                      desc="Inferencing",
                      disable=args.local_rank not in [-1,
                                                      0],
                      position=0,
                      leave=True):

        idxs = batch[3].detach().numpy()  # [#B]

        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0].long(),
                "attention_mask": batch[1].long()}
            if is_query_inference:
                embs = model.module.query_emb(**inputs)
            else:
                embs = model.module.body_emb(**inputs)

        embs = embs.detach().cpu().numpy()

        # check for multi chunk output for long sequence
        if len(embs.shape) == 3:
            for chunk_no in range(embs.shape[1]):
                embedding2id.append(idxs)
                embedding.append(embs[:, chunk_no, :])
        else:
            embedding2id.append(idxs)
            embedding.append(embs)

    embedding = np.concatenate(embedding, axis=0)
    embedding2id = np.concatenate(embedding2id, axis=0)
   
    return embedding, embedding2id


# streaming inference
def StreamInferenceDoc(args, model, fn, prefix, f, is_query_inference=True, load_from_public=False, only_load_in_master=True):
    inference_batch_size = args.per_gpu_eval_batch_size  # * max(1, args.n_gpu)
    inference_dataset = StreamingDataset(f, fn)
    inference_dataloader = DataLoader(
        inference_dataset,
        batch_size=inference_batch_size)

    if args.local_rank != -1:
        dist.barrier()  # directory created
    
    _embedding, _embedding2id = InferenceEmbeddingFromStreamDataLoader(
        args, model, inference_dataloader, is_query_inference=is_query_inference, prefix=prefix)     
    if args.local_rank != -1:
        dist.barrier()  # directory created, wait until all process finish
    logger.info("merging embeddings, %s"%(args.rank))

    # preserve to memory
    full_embedding = barrier_array_merge(
        args,
        _embedding,
        prefix=prefix +
        "_emb_p_",
        load_cache=False,
        only_load_in_master=only_load_in_master,
        load_from_public=load_from_public)
    full_embedding2id = barrier_array_merge(
        args,
        _embedding2id,
        prefix=prefix +
        "_embid_p_",
        load_cache=False,
        only_load_in_master=only_load_in_master,
        load_from_public=load_from_public)
    return full_embedding, full_embedding2id, None

def generate_new_ann(
        args,
        output_num,
        checkpoint_path,
        training_query_positive_id,
        dev_query_positive_id,
        latest_step_num):
    import random
    config, tokenizer, model = load_model(args, checkpoint_path)

    if latest_step_num != 0:
        ckpt_path = os.path.join(args.output_dir, 'msmarco')
        load_from_public = False
    else:
        ckpt_path = os.path.join(args.public_ann_data_dir, 'msmarco')
        load_from_public = True

    logger.info("***** inference of dev query *****")
    if embedding_dir_exist(checkpoint_path = ckpt_path, checkpoint = latest_step_num, groupid = False):
        logger.info("Path Exists! Loading from Existing Marco Embeddings")
        if is_first_worker():
            dev_query_embedding, dev_query_embedding2id, passage_embedding, passage_embedding2id, groupid = load_embedding(checkpoint_path = ckpt_path, checkpoint = latest_step_num,  groupid = bool(args.group), args = args)
    else:
        if is_first_worker():
            marco_dir = ckpt_path # os.path.join(args.output_dir, 'msmarco')
            try:
                os.makedirs(marco_dir, exist_ok = True)
            except OSError:
                print(f'Directory {marco_dir} cannot be created')

        dev_query_collection_path = os.path.join(args.data_dir, "dev-query")
        dev_query_cache = EmbeddingCache(dev_query_collection_path, group = False)

        with dev_query_cache as emb:
            dev_query_embedding, dev_query_embedding2id, _ = StreamInferenceDoc(args, model, GetProcessingFn(
                args, query=True), "msmarco/dev_query_" + str(latest_step_num) + "_", emb, is_query_inference=True, load_from_public = load_from_public)
        logger.info("***** inference of passages *****")
        passage_collection_path = os.path.join(args.data_dir, "passages")
        passage_cache = EmbeddingCache(passage_collection_path, group = False)
        with passage_cache as emb:
            passage_embedding, passage_embedding2id, _ = StreamInferenceDoc(args, model, GetProcessingFn(
                args, query=False), "msmarco/passage_" + str(latest_step_num) + "_", emb, is_query_inference=False, load_from_public = load_from_public)
    logger.info("***** Done passage inference *****")
    if args.local_rank != -1:
        dist.barrier()  # directory created

    if args.inference:
        return

    logger.info("***** inference of train query *****")
    train_query_collection_path = os.path.join(args.data_dir, "train-query")
    train_query_cache = EmbeddingCache(train_query_collection_path, group = False)
    with train_query_cache as emb:
        query_embedding, query_embedding2id, _ = StreamInferenceDoc(args, model, GetProcessingFn(
            args, query=True), "msmarco/query_" + str(latest_step_num) + "_", emb, is_query_inference=True, only_load_in_master = False, load_from_public = load_from_public)
    if is_first_worker():
        dim = passage_embedding.shape[1]
        print('passage embedding shape: ' + str(passage_embedding.shape))
        top_k = args.topk_training
        faiss.omp_set_num_threads(32)
        cpu_index = faiss.IndexFlatIP(dim)
        cpu_index.add(passage_embedding)
        logger.info("***** Done ANN Index *****")

        # measure ANN mrr
        # I: [number of queries, topk]
        _, dev_I = cpu_index.search(dev_query_embedding, 100)
        dev_ndcg, num_queries_dev, dev_mrr = EvalDevQuery(
            args, dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I)

        try:
            result_marco_dir = os.path.join(args.result_dir, 'msmarco')
            os.makedirs(result_marco_dir, exist_ok = True)
        except OSError:
            print(f'Directory {result_marco_dir} cannot be created')
        try:
            os.makedirs(os.path.join(args.output_dir, 'msmarco'), exist_ok = True)
        except OSError:
            print(f'Directory msmarco cannot be created')

        ####### Construct new training set =============================
        chunk_factor = args.ann_chunk_factor
        effective_idx = output_num % chunk_factor

        if chunk_factor <= 0:
            chunk_factor = 1
        num_queries = len(query_embedding)
        queries_per_chunk = num_queries // chunk_factor
        ################################################################
        if args.cluster_query:
            # use K-Means to generate querys for optimization
            dim = query_embedding.shape[-1]
            kmeans = faiss.Kmeans(dim, args.cluster_centroids, nredo=5, niter=500, verbose=True)
            # cluster using K-means with query embeddings
            q_embedding = query_embedding
            kmeans.train(q_embedding)
            index = faiss.IndexFlatL2(dim)
            index.add(kmeans.centroids )

            # D, I: matrix: kmeans_ncentroids * kmeans_topk
            topk =  queries_per_chunk // args.cluster_centroids
            D, I = index.search(q_embedding, 1)
            class_id = I.reshape(-1)
            # hard_sample
            cluster_cnt = defaultdict(list)
            cluster_sample_num = defaultdict()
            id_to_cluster =  defaultdict()
            cluster_id_ratio = defaultdict()
            # cluster_mrr = defaultdict(list)
            for i, cls_id in enumerate(class_id):
                cluster_cnt[cls_id].append(i)
                id_to_cluster[i] = int(cls_id)
            max_num = 0
            for cls in cluster_cnt:
                cluster_sample_num[cls] = len(cluster_cnt[cls])
                max_num = max(max_num, cluster_sample_num[cls])
            # cluster_id_ratio 
            for cls in cluster_sample_num:
                cluster_id_ratio[cls] = max(1.0, max_num / cluster_sample_num[cls])
            
            logger.info(f"Chunking {len(query_embedding)} Samples")
            query_embedding = query_embedding #[cluster_id]
            query_embedding2id = query_embedding2id #[cluster_id]

        else:
            q_start_idx = queries_per_chunk * effective_idx
            q_end_idx = num_queries if (
                effective_idx == (
                    chunk_factor -
                    1)) else (
                q_start_idx +
                queries_per_chunk)
            
            query_embedding = query_embedding[q_start_idx:q_end_idx]
            query_embedding2id = query_embedding2id[q_start_idx:q_end_idx]
            logger.info(f"Chunking {q_end_idx-q_start_idx+1} Samples")
        ################################################################# 
        train_data_output_path = os.path.join(
            args.output_dir, "msmarco/ann_training_data_" + str(output_num))
        _, I = cpu_index.search(query_embedding, top_k)

        effective_q_id = set(query_embedding2id.flatten())
        query_negative_passage, mrr_scores = GenerateNegativePassaageID(
            args,
            query_embedding2id,
            passage_embedding2id,
            training_query_positive_id,
            I,
            effective_q_id)

        logger.info("***** Construct ANN Triplet *****")

        with open(train_data_output_path, 'w') as f:
            query_range = list(range(I.shape[0]))
            random.shuffle(query_range)
            for split in range(5):
                for query_idx in tqdm(query_range):
                    query_id = query_embedding2id[query_idx]
                    mrr_score = float(mrr_scores[query_idx]) 
                    if query_id not in effective_q_id or query_id not in training_query_positive_id:
                        continue
                    pos_pid = training_query_positive_id[query_id]
                    neg_id_len = len(query_negative_passage[query_id])
                    start_id = split * (neg_id_len//5)
                    end_id = (1 + split) * (neg_id_len//5)
                    if args.cluster_query:                        
                        cluster_idx = id_to_cluster[query_idx]
                        weight = 1
                        f.write(
                            "{}\t{}\t{}\t{:.4f}\t{}\n".format(
                                query_id, pos_pid, ','.join(
                                    str(neg_pid) for neg_pid in query_negative_passage[query_id][start_id: end_id]), weight, int(cluster_idx)
                            ))
                    else:
                        f.write(
                            "{}\t{}\t{}\n".format(
                                query_id, pos_pid, ','.join(
                                    str(neg_pid) for neg_pid in query_negative_passage[query_id][start_id: end_id])
                            ))

        ndcg_output_path = os.path.join(
            args.result_dir, "msmarco/ann_ndcg_" + str(output_num))
        with open(ndcg_output_path, 'w') as f:
            json.dump({'ndcg': dev_ndcg, 'mrr': dev_mrr, 'checkpoint': checkpoint_path}, f)
        
        return dev_ndcg, num_queries_dev, dev_mrr

def embedding_dir_exist(checkpoint_path, checkpoint, groupid = False):
    '''
        checkpoint_path: model_name + '/' + dataset (for BEIR)
        checkpoint: id of the latest steps
    '''
    # print(checkpoint_path, os.path.join(checkpoint_path, "dev_query_" + str(checkpoint) + "__emb_p__data_obj_0.pb"), os.path.exists(os.path.join(checkpoint_path, "dev_query_" + str(checkpoint) + "__emb_p__data_obj_0.pb"))
    # exit(0)
    if os.path.exists(os.path.join(checkpoint_path, "dev_query_" + str(checkpoint) + "__emb_p__data_obj_0.pb")) and \
        os.path.exists(os.path.join(checkpoint_path, "dev_query_" + str(checkpoint) + "__embid_p__data_obj_0.pb")) and \
        os.path.exists(os.path.join(checkpoint_path, "passage_" + str(checkpoint) + "__emb_p__data_obj_0.pb")) and \
        os.path.exists(os.path.join(checkpoint_path, "passage_" + str(checkpoint) + "__embid_p__data_obj_0.pb")):
        if not groupid:        
            return True
        else:
            if os.path.exists(os.path.join(checkpoint_path, "dev_query_"+str(checkpoint) + "__groupid_p__data_obj_0.pb")):
                return True
            else:
                return False
    else:
        return False


def load_embedding(checkpoint_path, checkpoint, groupid = False, args = None):
    dev_query_embedding = []
    dev_query_embedding2id = []
    passage_embedding = []
    passage_embedding2id = []
    group2id = []

    for i in range(args.world_size):
        try:
            with open(checkpoint_path + "/dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                dev_query_embedding.append(pickle.load(handle))
            with open(checkpoint_path + "/dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                dev_query_embedding2id.append(pickle.load(handle))
            with open(checkpoint_path + "/passage_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                passage_embedding.append(pickle.load(handle))
            with open(checkpoint_path + "/passage_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                passage_embedding2id.append(pickle.load(handle))
            if groupid:
                with open(checkpoint_path + "/dev_query_"+str(checkpoint)+"__groupid_p__data_obj_"+str(i)+".pb", 'rb') as handle:
                    group2id.append(pickle.load(handle))
        except:
            break
    if (not dev_query_embedding) or (not dev_query_embedding2id) or (not passage_embedding) or not (passage_embedding2id):
        print("No data found for checkpoint: ",checkpoint)

    dev_query_embedding = np.concatenate(dev_query_embedding, axis=0)
    dev_query_embedding2id = np.concatenate(dev_query_embedding2id, axis=0)
    passage_embedding = np.concatenate(passage_embedding, axis=0)
    passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)
    if groupid:
        group2id = np.concatenate(group2id, axis=0)
    print('Loading Successful. Shape of Query Embedding:', dev_query_embedding.shape, 'Shape of Passage Embedding:', passage_embedding.shape)
    if groupid:
        return dev_query_embedding, dev_query_embedding2id, passage_embedding, passage_embedding2id, group2id
    else:
        return dev_query_embedding, dev_query_embedding2id, passage_embedding, passage_embedding2id, None

def GenerateNegativePassaageID(
        args,
        query_embedding2id,
        passage_embedding2id,
        training_query_positive_id,
        I_nearest_neighbor,
        effective_q_id):
    query_negative_passage = {}
    SelectTopK = args.ann_measure_topk_mrr
    mrr = 0  # only meaningful if it is SelectTopK = True
    num_queries = 0
    mrr_scores = []
    cnt = 0

    for query_idx in trange(I_nearest_neighbor.shape[0]):

        query_id = query_embedding2id[query_idx]

        if query_id not in effective_q_id:
            continue

        num_queries += 1

        pos_pid = training_query_positive_id[query_id]
        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()

        rank = 0
        in_top_K = 0
        for idx in top_ann_pid:
            neg_pid = passage_embedding2id[idx]
            rank += 1
            if neg_pid == pos_pid:
                mrr_scores.append(float(1.0/rank))
                cnt += 1
                in_top_K = 1
                break
        if in_top_K == 0:
            mrr_scores.append(0)

        if SelectTopK:
            selected_ann_idx = top_ann_pid[:args.negative_sample + 1]
        else:
            negative_sample_I_idx = list(range(I_nearest_neighbor.shape[1]))
            random.shuffle(negative_sample_I_idx)
            selected_ann_idx = top_ann_pid[negative_sample_I_idx]

        query_negative_passage[query_id] = []

        neg_cnt = 0
        rank = 0

        for idx in selected_ann_idx:
            neg_pid = passage_embedding2id[idx]
            rank += 1
            if neg_pid == pos_pid:
                if rank <= 10:
                    mrr += 1 / rank
                continue

            if neg_pid in query_negative_passage[query_id]:
                continue

            if neg_cnt >= args.negative_sample:
                break

            query_negative_passage[query_id].append(neg_pid)
            neg_cnt += 1

    if SelectTopK:
        print("Rank:" + str(args.rank) +
              " --- ANN MRR:" + str(mrr / num_queries))
    print(len(mrr_scores), I_nearest_neighbor.shape[0])

    return query_negative_passage, np.array(mrr_scores)


def EvalDevQuery(
        args,
        query_embedding2id,
        passage_embedding2id,
        dev_query_positive_id,
        I_nearest_neighbor,
        offset2qchar = None,
        offset2pchar = None):
    # [qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)
    prediction = {}

    for query_idx in range(I_nearest_neighbor.shape[0]):
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()
        selected_ann_idx = top_ann_pid[:100]
        rank = 0
        seen_pid = set()
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]

            if pred_pid not in seen_pid:
                # this check handles multiple vector per document
                rank += 1
                if offset2qchar and offset2qchar and query_id in offset2qchar and pred_pid in offset2pchar and  offset2pchar[pred_pid] == offset2qchar[query_id]:
                    # print('pass')
                    continue
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut', 'recip_rank'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))
    ndcg = 0
    mrr = 0

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]
        mrr += result[k]["recip_rank"]
    final_mrr = mrr / eval_query_cnt
    final_ndcg = ndcg / eval_query_cnt
    print("Rank:" + str(args.rank) + " --- ANN NDCG@10:" + str(final_ndcg) + " --- ANN MRR:" + str(final_mrr))

    return final_ndcg, eval_query_cnt, final_mrr

def EvalDevQueryforGroup(
        args,
        query_embedding2id,
        passage_embedding2id,
        dev_query_positive_id,
        I_nearest_neighbor,
        group2id):
    # [qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)
    prediction = {}

    for query_idx in range(I_nearest_neighbor.shape[0]):
        group_id = group2id[query_idx]
        if group_id not in prediction:
            prediction[group_id] = {}

        query_id = query_embedding2id[query_idx]
        prediction[group_id][query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx, :].copy()
        selected_ann_idx = top_ann_pid[:50]
        rank = 0
        seen_pid = set()
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]

            if pred_pid not in seen_pid:
                # this check handles multiple vector per document
                rank += 1
                prediction[group_id][query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut', 'recip_rank'})

    result_dict = {}
    for i in prediction:
        eval_query_cnt = 0
        result = evaluator.evaluate(convert_to_string_id(prediction[i]))
        ndcg = 0
        mrr = 0

        for k in result.keys():
            eval_query_cnt += 1
            ndcg += result[k]["ndcg_cut_10"]
            mrr += result[k]["recip_rank"]
        final_mrr = mrr / eval_query_cnt
        final_ndcg = ndcg / eval_query_cnt
        result_dict[str(int(i))] = {"final_mrr": final_mrr, "final_ndcg": final_ndcg, "eval_query_cnt": eval_query_cnt}
        print("Group:" + str(int(i)) + " Rank:" + str(args.rank) + " --- ANN NDCG@10:" + str(final_ndcg) + " --- ANN MRR:" + str(final_mrr))

    return result_dict #final_ndcg, eval_query_cnt, final_mrr

def get_arguments():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
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
        help="Initial model dir, will use this if no checkpoint is found in model_dir",
    )

    parser.add_argument(
        "--last_checkpoint_dir",
        default="",
        type=str,
        help="Last checkpoint used, this is for rerunning this script when some ann data is already generated",
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
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )

    parser.add_argument(
        "--public_ann_data_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory for public file (ALL exp will use it)",
    )

    parser.add_argument(
        "--result_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the training data will be written",
    )

    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        required=True,
        help="The directory where cached data will be written",
    )

    parser.add_argument(
        "--end_output_num",
        default=-1,
        type=int,
        help="Stop after this number of data versions has been generated, default run forever",
    )
    # parser.add_argument(
    #     "--max_seq_length_beir",
    #     default=512,
    #     type=int,
    #     help="The maximum total input sequence length after tokenization. Sequences longer "
    #     "than this will be truncated, sequences shorter will be padded.",
    # )

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
        "--max_doc_character",
        default=10000,
        type=int,
        help="used before tokenizer to save tokenizer latency",
    )

    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=128,
        type=int,
        help="The starting output file number",
    )

    parser.add_argument(
        "--ann_chunk_factor",
        default=1,  # for 500k queryes, divided into 100k chunks for each epoch
        type=int,
        help="devide training queries into chunks",
    )

    parser.add_argument(
        "--topk_training",
        default=500,
        type=int,
        help="top k from which negative samples are collected",
    )

    parser.add_argument(
        "--negative_sample",
        default=5,
        type=int,
        help="at each resample, how many negative samples per query do I use",
    )

    parser.add_argument(
        "--ann_measure_topk_mrr",
        default=False,
        action="store_true",
    )

    parser.add_argument(
        "--only_keep_latest_embedding_file",
        default=False,
        action="store_true",
        help="whether only keep the latest embeddings",
    )

    parser.add_argument(
        "--cluster_query",
        default=False,
        action="store_true",
        help="clustering based on queries or not",
    )

    parser.add_argument(
        "--rewei",
        default='',
        type=str,
        help="whether reweight the sample or not",
    )

    parser.add_argument(
        "--cluster_centroids",
        default=100,
        type=int,
        help="number of clusters",
    )

    parser.add_argument(
        "--no_cuda",
        action="store_true",
        help="Avoid using CUDA when available",
    )
    
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
        "--inference",
        default=False,
        action="store_true",
        help="only do inference if specify",
    )

    parser.add_argument(
        "--model_name",
        default='ance',
        type=str,
    )
    
    parser.add_argument(
        "--group",
        default=0,
        type=int,
    )

    args = parser.parse_args()

    return args


def set_env(args):
    args.rank =  int(dict(os.environ)["RANK"])
    args.server_ip = dict(os.environ)["MASTER_ADDR"]
    args.server_port = dict(os.environ)["MASTER_PORT"]
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.cuda.set_device(args.local_rank)
        # device = torch.device("cuda", args.local_rank)
        # torch.distributed.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        from torch._C._distributed_c10d import _DEFAULT_PG_TIMEOUT
        from datetime import timedelta
        addition_time = timedelta(minutes=110)
        total_timeout = _DEFAULT_PG_TIMEOUT + addition_time
        torch.distributed.init_process_group(backend="nccl", timeout = total_timeout, rank=args.rank)
        # print(dict(os.environ))
        # exit()
        args.n_gpu = 1
    args.device = device

    # store args
    if args.local_rank != -1:
        args.world_size = torch.distributed.get_world_size()
        args.rank = dist.get_rank()
        # print(args.world_size, "\t", args.rank, "\t", dist.get_rank())
        # exit()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
    )


def ann_data_gen(args):
    last_checkpoint = args.last_checkpoint_dir
    try:
        ann_no, ann_path, ndcg_json = get_latest_ann_data(args.output_dir, result_dir = args.result_dir, dataset = args.beir_dataset)
    except ValueError:
        ann_no, ann_path, ndcg_json,_ = get_latest_ann_data(args.output_dir, result_dir = args.result_dir, dataset = args.beir_dataset)

    output_num = ann_no + 1
    print("starting output number %d"%output_num)
    logger.info("starting output number %d", output_num)

    if is_first_worker():
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(args.cache_dir):
            os.makedirs(args.cache_dir)

    training_positive_id, dev_positive_id = load_positive_ids(args)

    # while args.end_output_num == -1 or output_num <= args.end_output_num:
    next_checkpoint, latest_step_num = get_latest_checkpoint(args)

    if args.only_keep_latest_embedding_file:
        latest_step_num = 0

    print(f"Loading Checkpoint of step {latest_step_num}")
    logger.info("start generate ann data number %d", output_num)
    logger.info("next checkpoint at " + next_checkpoint)
    generate_new_ann(
        args,
        output_num,
        next_checkpoint,
        training_positive_id,
        dev_positive_id,
        latest_step_num)
        
    # if args.inference:
        # break
    logger.info("finished generating ann data number %d", output_num)
    output_num += 1
    last_checkpoint = next_checkpoint
    if args.local_rank != -1:
        dist.barrier()


def main():
    args = get_arguments()
    set_env(args)
    ann_data_gen(args)


if __name__ == "__main__":
    import setproctitle
    setproctitle.setproctitle('ANCE_ann_data_gen@yuyue')
    main()
