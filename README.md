# COCO-DR
[EMNLP 2022] This repo provides the code for reproducing the experiments in paper **COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval with Contrastive and Distributionally Robust Learning** (EMNLP 2022 Main Conference).

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
- For additional packages, please run the following commands: `this need to be updated in a later version`
<!-- ```setup
git clone -b [branch_name] https://github.com/xiongchenyan/ZeroDR
cd ZeroDR
python setup.py install
``` -->

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
The code for reproducing COCO pretraining is in the `COCO` folder. Please checkout the `COCO/README.md` for detailed instructions. Note that we start COCO pretraining from the `condenser` checkpoint. We provide the `condenser` checkpoint for BERT Large as the backbone at [this link]().

### b. Finetuning with iDRO
- BM25 Warmup
	- The code for BM25 warmup is in the `warmup` folder.
- Training with global hard negative (ANCE):
	- The code for ANCE fine-tuning is in the `ANCE` folder.
  
## Checkpoints
We release the following checkpoints for both `COCO-DR Base` and `COCO-DR Large` to facilitate future studies:
- Pretrained model after COCO step w/o finetuning on MS MARCO.
- Pretrained model after iDRO step.
- Pretrained model after iDRO step (but w/o COCO): [This Link]() [Note: this model is trained *without* any BEIR task information].

|    Zero-shot Performance    |  Avg. on BEIR | Link |
|---------------- | -------------- | -------------- | 
| COCO-DR Base  |      0.462   |       [This Link]()       |
| COCO-DR Base (w/o COCO)  |         0.447       |  [This Link]()     |
| COCO-DR Base (w/o Finetuning on MS MARCO) |   0.289    |        |       [This Link]()       |
| COCO-DR Large   |       0.484       |  [This Link]()     |
| COCO-DR Large (w/o COCO)  |        0.463       |  [This Link]()     |
| COCO-DR Large (w/o Finetuning on MS MARCO) |  0.317     |        |       [This Link]()       |


## Citation
If you find this repository helpful, feel free to cite our publication [COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval with Contrastive and Distributional Robust Learning](404) (Preprint coming out soon!)

```
@inproceedings{yu2022cocodr,
  title={COCO-DR: Combating Distribution Shifts in Zero-Shot Dense Retrieval with Contrastive and Distributionally Robust Learning},
  author={Yue Yu and Chenyan Xiong and Si Sun and Chao Zhang and Arnold Overwijk},
  booktitle={Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  year={2022}
}
```
