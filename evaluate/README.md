# Evaluating on BEIR Tasks
This folder contains the code for evaluating the models on the BEIR benchmark.

## Dependencies
The code uses the following packages
```
pandas
transformers==2.3.0 
pytrec_eval 
faiss-cpu==1.6.4 
wget 
scikit-learn 
urllib3 
requests 
jmespath 
tqdm 
setproctitle
tables
```

## Evaluating Steps
### Step 1: Tokenizing Text Sequence

An example for using the script for tokenizing the dataset is shown as belows:
```
python ../data/beir_data.py \
--data_dir ${base_data_dir}${dataset} \
--out_data_dir $preprocessed_data_dir \
--model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir \
--max_query_length $max_query_length \
--max_seq_length $max_seq_length \
--data_type $data_type \
--dataset ${dataset}
```

Here 
+ `base_data_dir` is the folder for saving BEIR datasets;
```
  ├── base_data_dir/ -- base directory for storing the data.
    ├── nfcorpus
    ├── scifact
    ├── scidocs
    └── ....
```
For each dataset, it has a folder including both query file, corpus file and qrels file. The format of the dataset is shown in [this link](https://github.com/beir-cellar/beir/wiki/Load-your-custom-dataset).
+ `preprocessed_data_dir`: the folder for saving the processed datasets.
+ `model_type`: the type of encoder. We use `rdot_nll_condenser` in  our experiments.
+ `pretrained_checkpoint_dir`: the directory of the model checkpoint.
+ `max_query_length`: the maximum length for queries (128 for `Arguana` and 64 for other tasks).
+ `max_seq_length`: the maximum length for documents (256 for `TREC-NEWS, Robust04, SciFact` and 128 for other tasks).
+ `data_type`: the type of text. We set it to `1` in our experiments.
+ `dataset`: the name of dataset for evaluation.

The tokenized files will be saved to `preprocessed_data_dir`.

### Step 2: Generating Embeddings for Queries and Documents.
An example for using the script for generating emebddings is shown as belows:
```
python -m torch.distributed.launch --nproc_per_node=$gpu_no --master_port $port_id ../drivers/run_ann_data_gen.py \
--training_dir ${pretrained_checkpoint_dir} \
--model_type ${model_type} \
--output_dir ${model_ann_data_dir} \
--cache_dir "${model_ann_data_dir}cache/" \
--data_dir ${preprocessed_data_dir} \
--max_query_length=${max_query_length} \
--max_seq_length ${max_seq_length} \
--per_gpu_eval_batch_size 256 \
--end_output_num 0 \
--inference 
```

Here 
+ `gpu_no`: the number of GPU used for evaluation.
+ `port_id`: the id for port. Can be set to a random number (e.g. `44343`).
+ `pretrained_checkpoint_dir`: the directory of the model checkpoint.
+ `model_ann_data_dir`: the directory of the generated query/document embeddings.
+ `preprocessed_data_dir`:  the folder for saving the processed datasets (generated in the step 1).
+ `max_query_length`: the maximum length for queries.
+ `max_seq_length`: the maximum length for documents.


### Step 3: Evaluation on the target tasks
An example for calculating the nDCG@10 with the generated emebddings is shown as belows:
```
python ../evaluation/evaluate_beir.py --dataset=${dataset} \
--preprocessed_data_dir=${preprocessed_data_dir} \
--model_ann_data_dir=${model_ann_data_dir}"
```
Here
+ `preprocessed_data_dir`:  the folder for saving the processed datasets (generated in the step 1).
+ `model_ann_data_dir`: the directory of the generated query/document embeddings (generated in the step 2).

A sample script for evaluation can be find at `commands/run_evaluate.sh`.