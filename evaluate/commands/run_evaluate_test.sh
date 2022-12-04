Directory_of_BEIR_Data=/home/yueyu/beir/datasets/
Directory_of_Processed_Data=/home/yueyu/beir/datasets/process/
Directory_of_Model_Checkpoint=/home/yueyu/coco-dr-backup/backup/cocodr-base/after_ance
base_data_dir=${Directory_of_BEIR_Data}
dataset=hotpotqa
preprocessed_data_dir=${Directory_of_Processed_Data}
mkdir -p ${preprocessed_data_dir}
model_type=rdot_nll_condenser
pretrained_checkpoint_dir=${Directory_of_Model_Checkpoint}
max_query_length=64
max_seq_length=128
data_type=1


python ../data/beir_data.py \
--data_dir ${base_data_dir}${dataset} \
--out_data_dir ${preprocessed_data_dir}${dataset} \
--model_type $model_type \
--model_name_or_path $pretrained_checkpoint_dir \
--max_query_length $max_query_length \
--max_seq_length $max_seq_length \
--data_type $data_type \
--dataset ${dataset}


gpu_no=4
port_id=15331
model_ann_data_dir=/home/yueyu/beir/datasets/process/embeddings/
mkdir -p ${model_ann_data_dir}
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=$gpu_no --master_port $port_id ../drivers/run_ann_data_gen.py \
--training_dir ${pretrained_checkpoint_dir} \
--model_type ${model_type} \
--output_dir ${model_ann_data_dir} \
--cache_dir "${model_ann_data_dir}cache/" \
--data_dir ${preprocessed_data_dir}${dataset} \
--max_query_length=${max_query_length} \
--max_seq_length ${max_seq_length} \
--per_gpu_eval_batch_size 512 \
--end_output_num 0 \
--inference

python ../evaluation/evaluate_beir.py --dataset=${dataset} \
--preprocessed_data_dir=${preprocessed_data_dir}${dataset} \
--model_ann_data_dir=${model_ann_data_dir}