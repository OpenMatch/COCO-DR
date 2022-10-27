# This script is for training the warmup checkpoint for COCO-DR
gpu_no=8
seq_length=512
model_type=rdot_nll
tokenizer_type="roberta-base"
data_type=1
per_gpu_train_batch_size=256
gradient_accumulation_steps=1
learning_rate=1e-4
warmup_step=1000
max_steps=2000

DEVICE=0,1,2,3,4,5,6,7
ngpu=8


base_data_dir="YOUR_DIRECTORY_FOR_TRAINING_DATA"
model_name_or_path="YOUR_PATH_FOR_CHECKPOINT"

train_model_type='rdot_nll_condenser'

data_dir="${base_data_dir}msmarco"

train_data_dir="${base_data_dir}msmarco/triples.train.small.tsv"
expected_train_size=35000000

# directory for saving the checkpoints
output_dir="${base_data_dir}runs"

mkdir -p $output_dir
echo "Model save dir: $output_dir" 

TSB_OUTPUT_DIR="${base_data_dir}runs/$tsb"
mkdir -p $TSB_OUTPUT_DIR

common_cmd="--model_name_or_path ${model_name_or_path} \
  --task_name MSMarco --do_train --evaluate_during_training --data_dir ${data_dir}  --train_data_dir ${train_data_dir} \
  --max_seq_length 128     --per_gpu_eval_batch_size=128 \
  --per_gpu_train_batch_size=${per_gpu_train_batch_size}    --learning_rate ${learning_rate}  --logging_steps 1000   \
  --num_train_epochs 3.0  --output_dir ${output_dir} \
  --warmup_steps ${warmup_step}   --overwrite_output_dir --save_steps 5000 --gradient_accumulation_steps 1 \
  --expected_train_size ${expected_train_size} --logging_steps_per_eval 10 \
  --fp16 --optimizer lamb --log_dir ${TSB_OUTPUT_DIR}"

cmd="CUDA_VISIBLE_DEVICES=${DEVICE} python3 -m torch.distributed.launch --nproc_per_node=${ngpu}  --master_port=${PORT} ../drivers/run_bm25_warmup.py --train_model_type ${train_model_type} ${common_cmd}"

echo $cmd
eval $cmd 