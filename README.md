# COCO-DR
This repo provides the code for reproducing the experiments in paper **COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval with Contrastive and Distributionally Robust Learning** (EMNLP 2022 Main Conference).

COCO-DR is a domain adaptation method for training zero-shot dense retrievers. It is based on simple *continuous constrastive learning* (COCO) and *implicit distributional robust learning* (iDRO) and can achieve significant improvement over other zero-shot models without using billion-scale models, seq2seq models, and cross-encoder distillation.

## Quick Links

  - [BEIR Performance](#BEIR-Performance)
  - [Model Checkpoints](#Checkpoints)
  - [Using COCO-DR with Huggingface](#Usage)
  - [Train COCO-DR](#Experiments)
    - [COCO Pretraining](#COCO-Pretraining)
    - [iDRO Finetuning](#Finetuning-with-iDRO)
  - [Evaluating on BEIR](#Evaluation-on-BEIR)
  - [Bugs or Questions?](#bugs-or-questions)
  - [Citation](#citation)


## BEIR Performance

|   Model   | BM25 | DocT5query |  [GTR](https://arxiv.org/abs/2112.07899) | [CPT-text](https://arxiv.org/abs/2201.10005)  | [GPL](https://arxiv.org/abs/2112.07577) | COCO-DR Base | COCO-DR Large |
|----------------- | -------------- |-------------- | -------------- | -------------- | -------------- | -------------- | -------------- |  
|   # of Parameter   | --- | --- |  4.8B | 178B | 66M*18 | 110M | 335M   
|   Avg. on BEIR CPT sub | 0.484 | 0.495 | 0.516 | 0.528  | 0.516 | 0.520 | **0.540**
|   Avg. on BEIR   |  0.428 | 0.440 | 0.458 | ---  | 0.459 |  0.461 | **0.484**

Note: 
+ `GPL` trains a separate model for each task and use cross-encoders for distillation.
+ `CPT-text` evaluate only on 11 selected subsets of the BEIR benchmark.


## Experiment Setup
### Environment
- We use this docker image for all our experiments: `mmdog/pytorch:pytorch1.9.0-nccl2.9.9-cuda11.3`. 
- For additional packages, please run the following commands in folders.

### Datasets
We use BEIR corpora for the COCO step, and use `MS Marco` dataset in the iDRO step. The procedure for obtaining the datasets will be described as follows.

#### MS Marco
- We use the dataset from the same source as the [ANCE](https://github.com/microsoft/ANCE) paper. The commands for downloading the MS Marcodataset can be found in `commands/data_download.sh`

#### BEIR
- We use the dataset released by the original [BEIR repo](https://github.com/beir-cellar/beir/blob/main/README.md). It can be downloaded at [this link](https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets).
- Note that due to copyright restrictions, some datasets are not available.


## Experiments
To run the experiments, use the following commands:

### COCO Pretraining
The code for reproducing COCO pretraining is in the `COCO` folder. Please checkout the `COCO/README.md` for detailed instructions. Note that we start COCO pretraining from the `condenser` checkpoint. We release the `condenser` checkpoint using BERT Large as the backbone at [this link](https://huggingface.co/OpenMatch/condenser-large).

### Finetuning with iDRO
- BM25 Warmup
	- The code for BM25 warmup is in the [warmup](warmup) folder.
- Training with global hard negative (ANCE):
	- The code for ANCE fine-tuning is in the [ANCE](ANCE) folder. 
  
### Evaluation on BEIR
The code for evaluation on BEIR is in the [evaluate](evaluate) folder. 

## Checkpoints
### Main Experiments
We release the following checkpoints for both `COCO-DR Base` and `COCO-DR Large` to facilitate future studies:
- Pretrained model after COCO step w/o finetuning on MS MARCO.
- Pretrained model after iDRO step.
- Pretrained model after iDRO step (but w/o COCO). Note: this model is trained *without* any BEIR task information.

|    Model Name   |  Avg. on BEIR | Link |
|---------------- | -------------- | -------------- | 
| COCO-DR Base  |      0.461   |       [OpenMatch/cocodr-base-msmarco](https://huggingface.co/OpenMatch/cocodr-base-msmarco)       |
| COCO-DR Base (w/o COCO)  |         0.447       |  [OpenMatch/cocodr-base-msmarco-idro-only](https://huggingface.co/OpenMatch/cocodr-base-msmarco-idro-only)     |
| COCO-DR Base (w/ BM25 Warmup)  |         0.435       |  [OpenMatch/cocodr-base-msmarco-warmup](https://huggingface.co/OpenMatch/cocodr-base-msmarco-warmup)     |
| COCO-DR Base (w/o Finetuning on MS MARCO) |   0.288    |        [OpenMatch/cocodr-base](https://huggingface.co/OpenMatch/cocodr-base)       |
| COCO-DR Large   |       0.484       |  [OpenMatch/cocodr-large-msmarco](https://huggingface.co/OpenMatch/cocodr-large-msmarco)     |
| COCO-DR Large (w/o COCO)  |        0.462       |  [OpenMatch/cocodr-large-msmarco-idro-only](https://huggingface.co/OpenMatch/cocodr-large-msmarco-idro-only)     |
| COCO-DR Large (w/ BM25 Warmup)  |         0.456       |  [OpenMatch/cocodr-large-msmarco-warmup](https://huggingface.co/OpenMatch/cocodr-large-msmarco-warmup)     |
| COCO-DR Large (w/o Finetuning on MS MARCO) |  0.316      |       [OpenMatch/cocodr-large](https://huggingface.co/OpenMatch/cocodr-large)       |

**Note**: We find a mismatch between the version of HotpotQA dataset we use  and the HotpotQA dataset used in BEIR. We rerun the evaluation and  update the number for HotpotQA using the *latest* version in BEIR.

### Other Models
Besides, to ensure reproducibility (especially for BERT-large), we also provide checkpoints from some *important* baselines that are re-implemented by us.
 |    Model Name    |   Link |
|---------------- |  -------------- | 
| Condenser Large (w/o Finetuning on MS MARCO) |        [OpenMatch/condenser-large](https://huggingface.co/OpenMatch/condenser-large)       |
| coCondenser Large (w/o Finetuning on MS MARCO) |        [OpenMatch/co-condenser-large](https://huggingface.co/OpenMatch/co-condenser-large)       |
| coCondenser Large (Fine-tuned on MS MARCO) |        [OpenMatch/co-condenser-large-msmarco](https://huggingface.co/OpenMatch/co-condenser-large-msmarco)       |


## Usage

Pre-trained models can be loaded through the HuggingFace transformers library:

```python
from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("OpenMatch/cocodr-base-msmarco") 
tokenizer = AutoTokenizer.from_pretrained("OpenMatch/cocodr-base-msmarco") 
```

Then embeddings for different sentences can be obtained by doing the following:

```python

sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
embeddings = model(**inputs, output_hidden_states=True, return_dict=True).hidden_states[-1][:, :1].squeeze(1) # the embedding of the [CLS] token after the final layer
```

Then similarity scores between the different sentences are obtained with a dot product between the embeddings:
```python

score01 = embeddings[0] @ embeddings[1] # 216.9792
score02 = embeddings[0] @ embeddings[2] # 216.6684
```


## Bugs or questions?

If you have any questions related to the code or the paper, feel free to email Yue Yu (`yueyu` at `gatech` dot `edu`) or open an issue. Please try to specify the problem with details so we can help you better and quicker!

## Citation
If you find this repository helpful, feel free to cite our publication [COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval with Contrastive and Distributional Robust Learning](https://arxiv.org/abs/2210.15212). 

```
@inproceedings{yu2022cocodr,
  title={COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval with Contrastive and Distributionally Robust Learning},
  author={Yue Yu and Chenyan Xiong and Si Sun and Chao Zhang and Arnold Overwijk},
  booktitle={Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing},
  year={2022}
}
```

## Acknowledgement
We would like to thank the authors from [ANCE](https://github.com/microsoft/ANCE) and [Condenser](https://github.com/luyug/Condenser) for their open-source efforts.
