# COCO Pretraining
This folder contains the code for reproducing the pretraining step of the COCO-DR model. 

## Dependencies
The code uses the following packages,
```
pytorch
transformers
datasets
nltk
```

## Model Initialzation
We initialized our model using the `condenser` checkpoint. The link of the checkpoint can be found in [this link](https://boston.lti.cs.cmu.edu/luyug/condenser/condenser.tar.gz).

## Data Preparation
Download the MS Marco and BEIR corpus at [this link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/).


## COCO Pre-training
### Pre-processing
Before COCO pretraining, we need to tokenize all the training text. The pre-processor expects one training document per line, with document broken into spans, e.g.
```
{'spans': List[str]}
...
```

You can run `pre_processing_coco.sh` to preprocess the data. Note that MS Marco and BEIR corpora are supposed to save at `./BEIR` folder. The processed data are saved at `./preprocessed_corpus` folder.

### Pre-training
Launch training with the following script. Our experiments in the paper warm start the coCondenser (both head and backbone) from a Condenser checkpoint.
```
python -m torch.distributed.launch --nproc_per_node $NPROC run_coco_pre_training.py \
  --output_dir $OUTDIR \
  --model_name_or_path /path/to/pre-trained/condenser/model \
  --do_train \
  --save_steps 30000 \
  --model_type bert \
  --per_device_train_batch_size $BATCH_SIZE \
  --gradient_accumulation_steps 1 \
  --fp16 \
  --warmup_ratio 0.1 \
  --learning_rate 1e-4 \
  --num_train_epochs 8 \
  --dataloader_drop_last \
  --overwrite_output_dir \
  --dataloader_num_workers 32 \
  --n_head_layers 2 \
  --skip_from 6 \
  --max_seq_length $MAX_LENGTH \
  --train_dir $JSON_SAVE_DIR \
  --weight_decay 0.01 \
  --late_mlm
```
Having `NPROC x BATCH_SIZE` to be large is critical for effective contrastive pre-training. It is set to roughly 2048 in our experiments.
*Warning: gradient_accumulation_steps should be kept at 1 as accumulation cannot emulate large batch for contrative loss.*


## Fine-tuning
The saved model can be loaded directly using huggingface interface and fine-tuned,
```
from transformers import AutoModel
model = AutoModel.from_pretrained('path/to/train/output')
```
The head will then be automatically omitted in fine-tuning.

## Acknowledgement
This repo is adapted from the [Condenser repo](https://github.com/luyug/Condenser). We sincerely appreciate the authors from the condenser paper for the implementations. 