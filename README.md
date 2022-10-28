# COCO-DR
This repo provides the code for reproducing the experiments in paper **COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval with Contrastive and Distributionally Robust Learning** (EMNLP 2022 Main Conference).

COCO-DR is a domain adaptation method for training zero-shot dense retrievers. It is based on simple *continuous constrastive learning* (COCO) and *implicit distributional robust learning* (iDRO) and can achieve significant improvement over other zero-shot models without using billion-scale models, seq2seq models, and cross-encoder distillation.

## BEIR Performance

|   Model   | BM25 | DocT5query |  [GTR](https://arxiv.org/abs/2112.07899) | [CPT-text](https://arxiv.org/abs/2201.10005)  | [GPL](https://arxiv.org/abs/2112.07577) | COCO-DR Base | COCO-DR Large |
|----------------- | -------------- |-------------- | -------------- | -------------- | -------------- | -------------- | -------------- |  
|   # of Parameter   | --- | --- |  4.8B | 178B | 66M*18 | 110M | 335M   
|   Avg. on BEIR CPT sub | 0.484 | 0.495 | 0.516 | 0.528  | 0.516 | 0.521 | **0.541**
|   Avg. on BEIR   |  0.428 | 0.440 | 0.458 | ---  | 0.459 |  0.462 | **0.484**

Note: 
+ `GPL` trains a separate model for each task.
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

### a. COCO Pretraining
The code for reproducing COCO pretraining is in the `COCO` folder. Please checkout the `COCO/README.md` for detailed instructions. Note that we start COCO pretraining from the `condenser` checkpoint. We provide the `condenser` checkpoint for BERT Large as the backbone at [this link] (Coming Soon!).

### b. Finetuning with iDRO
- BM25 Warmup
	- The code for BM25 warmup is in the `warmup` folder.
- Training with global hard negative (ANCE):
	- The code for ANCE fine-tuning is in the `ANCE` folder. (Coming Soon!)
  
### c. Evaluation on BEIR
The code for evaluation on BEIR is in the `evaluation` folder (Coming Soon!).

## Checkpoints
### Main Experiments
We release the following checkpoints for both `COCO-DR Base` and `COCO-DR Large` to facilitate future studies:
- Pretrained model after COCO step w/o finetuning on MS MARCO.
- Pretrained model after iDRO step.
- Pretrained model after iDRO step (but w/o COCO). Note: this model is trained *without* any BEIR task information.

|    Model Name   |  Avg. on BEIR | Link |
|---------------- | -------------- | -------------- | 
| COCO-DR Base  |      0.462   |       [OpenMatch/cocodr-base-msmarco](https://huggingface.co/OpenMatch/cocodr-base-msmarco)       |
| COCO-DR Base (w/o COCO)  |         0.447       |  [OpenMatch/cocodr-base-idro-only](https://huggingface.co/OpenMatch/cocodr-base-idro-only)     |
| COCO-DR Base (w/ BM25 Warmup)  |         0.436       |  [OpenMatch/cocodr-base-msmarco-warmup](https://huggingface.co/OpenMatch/cocodr-base-msmarco-warmup)     |
| COCO-DR Base (w/o Finetuning on MS MARCO) |   0.289    |        [OpenMatch/cocodr-base](https://huggingface.co/OpenMatch/cocodr-base)       |
| COCO-DR Large   |       0.484       |  [OpenMatch/cocodr-large-msmarco](https://huggingface.co/OpenMatch/cocodr-large-msmarco)     |
| COCO-DR Large (w/o COCO)  |        0.463       |  [Coming Soon!]()     |
| COCO-DR Large (w/ BM25 Warmup)  |         0.457       |  [Coming Soon!]()     |
| COCO-DR Large (w/o Finetuning on MS MARCO) |  0.317      |       [OpenMatch/cocodr-large](https://huggingface.co/OpenMatch/cocodr-large)       |

### Other Models
Besides, to ensure reproducibility (especially for BERT-large), we also provide checkpoints from some *important* baselines that are re-implemented by us.
 |    Model Name    |   Link |
|---------------- |  -------------- | 
| Condenser-large (w/o Finetuning on MS MARCO) |        [OpenMatch/condenser-large](https://huggingface.co/OpenMatch/condenser-large)       |
| coCondenser-large (w/o Finetuning on MS MARCO) |        [OpenMatch/co-condenser-large](https://huggingface.co/OpenMatch/co-condenser-large)       |
| coCondenser-large (Fine-tuned on MS MARCO) |        [OpenMatch/co-condenser-large-msmarco](https://huggingface.co/OpenMatch/co-condenser-large-msmarco)       |



## Citation
If you find this repository helpful, feel free to cite our publication [COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval with Contrastive and Distributional Robust Learning](https://arxiv.org/abs/2210.15212)

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
