# ANCE Training with iDRO
This folder contains the code for `ANCE` training with the implicit Distributionally Robust Optimization (iDRO) stragegy. 

The command to install the required package is in `commands/install.sh`. (Note that there is some differences between the used package in `BM25 warmup` and `COCO Pretraining`.)

The command with our used parameters to train this warmup checkpoint is in `commands/run_ance.sh`. 


## ANCE Training

### Step 1: Tokenizing Text Sequence

An example for using the script for tokenizing the dataset is shown as belows:
*Note that we do not use title information*

```
python ../data/msmarco_data.py \
--data_dir ${training_data_dir}  \
--out_data_dir ${preprocessed_data_dir} \
--model_type $model_type \
--model_name_or_path ${pretrained_checkpoint_dir} \
--max_seq_length 256 \
--data_type 1
```

### Step 2: Hard Negative Mining

`drivers/run_ann_data_gen_beir.py` provides the code to generates the hard negatives from the current (latest) checkpoint. An example usage of the script is given below:
```
python -m torch.distributed.launch --nproc_per_node=8 --master_port 12345 ../drivers/run_ann_data_gen.py \
--training_dir $saved_models_dir \
--init_model_dir $pretrained_checkpoint_dir \
--model_type rdot_nll_condenser \
--output_dir $model_ann_data_dir \
--cache_dir "${model_ann_data_dir}cache/" \
--data_dir $preprocessed_data_dir \
--max_seq_length 256 \
--per_gpu_eval_batch_size 512 \
--topk_training 200 \
--negative_sample 30 \
--end_output_num 0 \
--result_dir ${result_dir} \
--group 0 \
--public_ann_data_dir=${public_ann_data_dir} \
--cluster_query \
--cluster_centroids 50
```

Here
- `training_dir` is the directory for saving model checkpoints during training. If this directory is empty, then the checkpoint from `init_model_dir` will be used.
- `output_dir` is the directory for saving the embedding file and the calculated hard negative samples.
- `result_dir` is the directory for saving the evalation results on *MS MARCO*. You probably need to use the code in the `evaluate` folder with the trained checkpoint to evaluate on BEIR tasks.
- `cluster_query` stands for clustering the training data (used in iDRO) based on the query embeddings.
- `cluster_centroids` is used for setting the number of the clusters.

### Step 3: Training with Mined Hard Negative and iDRO

```
python -m torch.distributed.launch --nproc_per_node=8  --master_port 21345 ../drivers/run_ann.py \
--model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir \
--task_name MSMarco \
--training_dir=${saved_models_dir} \
--init_model_dir=${pretrained_checkpoint_dir} \
--triplet \
--data_dir $preprocessed_data_dir \
--ann_dir $model_ann_data_dir \
--max_seq_length $seq_length \
--per_gpu_train_batch_size $per_gpu_train_batch_size \
--per_gpu_eval_batch_size 512 \
--gradient_accumulation_steps 1 \
--learning_rate $learning_rate \
--output_dir $saved_models_dir \
--warmup_steps $warmup_steps \
--logging_steps=1000 \
--save_steps 3000 \
--max_steps ${MAX_STEPS} \
--single_warmup \
--optimizer lamb \
--fp16 \
--log_dir $TSB_OUTPUT_DIR \
--model_size=${MODEL_SIZE} \
--result_dir=${result_dir} \
--group=${group} \
--n_groups=${CLUSTER_NUM} \
--dro_type=${DRO_TYPE} \
--alpha=${alpha} \
--eps=${eps} \
--ema=${ema} \
--rho=${rho} \
--round=${i}
```

Here
- `saved_models_dir` is the folder for saving the checkpoints during training.
- `dro_type` is the type of the DRO algorithm. We provide two implementations: `iDRO` (the main method for this work) and `dro-greedy` (the original DRO method in [this paper](https://arxiv.org/abs/1911.08731)).
- `model_size` is the size of the model (base/large).
- `round` is the current episode (0 stands for the 1st episode for ANCE).
- `alpha`, `ema`, `rho`, `eps` are hyperparameters for the DRO.

## Key Hyperparameters
| Hyperparameter | Value |
| -------------- | -------------- |
| Max Learning Rate |  5e-6 for base / large |
| Warmup Steps |  3000 / 3000 for base / large |
| Max Training Steps for Each Episode |  45000 / 30000 for base / large|
| Batch Size per GPU |  64 / 32 for base / large|
| alpha |  0.25 for base / large |
| ema |  0.1 for base / large |
| rho |  0.05 for base / large |
| eps | 0.01 for base / large |
