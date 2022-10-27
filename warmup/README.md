# BM25 Warmup
This folder contains the code for `BM25 warmup`. 

The command to install the required package is in `commands/install.sh`. (Note that there is some differences between the used package in `BM25 warmup` and `COCO Pretraining`.)

The command with our used parameters to train this warmup checkpoint is in `commands/run_bm25_warmup.sh`. 

## Warmup Training
`drivers/run_bm25_warmup.py` provides the code to train a model. An example usage of the script is given below:
```
python3 -m torch.distributed.launch --nproc_per_node=1 ../drivers/run_warmup.py \
  --model_name_or_path ${your_model_name_or_path} \
  --task_name MSMarco \
  --do_train  \
  --evaluate_during_training   \
  --data_dir ${train_data_dir}  \
  --train_data_dir ${the_directory_of_file_used_for_training} \
  --max_seq_length 128  \
  --per_gpu_eval_batch_size=128 \
  --per_gpu_train_batch_size=${per_gpu_train_batch_size}  \
  --learning_rate 2e-4 \ 
  --logging_steps 10000 \
  --num_train_epochs 3.0 \ 
  --output_dir ${output_dir} \
  --warmup_steps ${warmup_step} \
  --overwrite_output_dir \
  --save_steps 5000 \
  --gradient_accumulation_steps 1 \
  --expected_train_size 35000000 \
  --logging_steps_per_eval 10 \
  --fp16 \
  --optimizer lamb \
  --log_dir ${TSB_OUTPUT_DIR}
```

## Key Hyperparameter
| Hyperparameter | Value |
| -------------- | -------------- |
| Max Learning Rate |  2e-4 / 5e-5 for base / large|
| Warmup Steps |  1000 / 5000 for base / large|
| Batch Size per GPU |  256 / 64 for base / large|
