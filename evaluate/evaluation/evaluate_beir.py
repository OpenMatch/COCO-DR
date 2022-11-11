import sys
sys.path += ['../utils']
import csv
from tqdm import tqdm 
import collections
import gzip
import pickle
import numpy as np
import faiss
import os
import pytrec_eval
import json
from msmarco_eval import quality_checks_qids, compute_metrics, load_reference
import argparse, pickle

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset',
        default=None,
        type=str,
        required=True,
        help= 'The name of the dataset'
    )
    parser.add_argument(
        '--model_name',
        default='ance',
        type=str,
        help= 'The name of the model'
    )

    parser.add_argument(
        '--preprocessed_data_dir',
        default='',
        type=str,
        help= 'The directory of the processed dataset'
    )

    parser.add_argument(
        '--model_ann_data_dir',
        default='',
        type=str,
        help= 'The directory of the generated embeddings'
    )
    args = parser.parse_args()
    return args

args = get_arguments()
dataset = args.dataset
checkpoint_path = args.model_ann_data_dir
checkpoint =  0 # embedding from which checkpoint(ie: 200000)
data_type = 1 # 0 for document, 1 for passage
test_set = 1 # 0 for dev_set, 1 for eval_set

#### newly added ####
processed_data_dir = args.preprocessed_data_dir
#####################

if data_type == 0:
    topN = 100
else:
    topN = 1000
dev_query_positive_id = {}
query_positive_id_path = os.path.join(processed_data_dir, "dev-qrel.tsv")


with open(query_positive_id_path, 'r', encoding='utf8') as f:
    tsvreader = csv.reader(f, delimiter="\t")
    for [topicid, docid, rel] in tsvreader:
        topicid = int(topicid)
        # topicid = offset2qid[topicid]
        docid = int(docid)
        # docid = offset2pid[docid]
        if topicid not in dev_query_positive_id:
            dev_query_positive_id[topicid] = {}
        dev_query_positive_id[topicid][docid] = max(0, int(float(rel))) #if int(float(rel)) > 0 else 0


pid2offset = pickle.load(open(os.path.join(processed_data_dir, "pid2offset.pickle"), 'rb'))
pchar2pid = pickle.load(open(os.path.join(processed_data_dir, "pchar2pid.pickle"), 'rb'))

qid2offset = pickle.load(open(os.path.join(processed_data_dir, "qid2offset.pickle"), 'rb'))
qchar2qid = pickle.load(open(os.path.join(processed_data_dir, "qchar2qid.pickle"), 'rb'))
for x in qchar2qid:
    y = qchar2qid[x]
    z = pid2offset[y]
offset2pchar = {pid2offset[pchar2pid[x]]:x for x in pchar2pid if pchar2pid[x] in pid2offset}
    
offset2qchar = {qid2offset[qchar2qid[x]]:x for x in qchar2qid if qchar2qid[x] in qid2offset}


def convert_to_string_id(result_dict):
    string_id_dict = {}

    # format [string, dict[string, val]]
    for k, v in result_dict.items():
        _temp_v = {}
        for inner_k, inner_v in v.items():
            _temp_v[str(inner_k)] = inner_v

        string_id_dict[str(k)] = _temp_v

    return string_id_dict

def EvalDevQuery(query_embedding2id, passage_embedding2id, dev_query_positive_id, I_nearest_neighbor,topN):
    prediction = {} #[qid][docid] = docscore, here we use -rank as score, so the higher the rank (1 > 2), the higher the score (-1 > -2)

    total = 0
    labeled = 0
    Atotal = 0
    Alabeled = 0
    qids_to_ranked_candidate_passages = {} 
    for query_idx in range(len(I_nearest_neighbor)): 
        seen_pid = set()
        query_id = query_embedding2id[query_idx]
        prediction[query_id] = {}

        top_ann_pid = I_nearest_neighbor[query_idx].copy()
        selected_ann_idx = top_ann_pid[:topN]
        rank = 0
        
        if query_id in qids_to_ranked_candidate_passages:
            pass    
        else:
            # By default, all PIDs in the list of 1000 are 0. Only override those that are given
            tmp = [0] * 1000
            qids_to_ranked_candidate_passages[query_id] = tmp
                
        for idx in selected_ann_idx:
            pred_pid = passage_embedding2id[idx]
            
            if not pred_pid in seen_pid:
                # this check handles multiple vector per document
                qids_to_ranked_candidate_passages[query_id][rank]=pred_pid
                Atotal += 1
                if pred_pid not in dev_query_positive_id[query_id]:
                    Alabeled += 1
                if rank < 10:
                    total += 1
                    if pred_pid not in dev_query_positive_id[query_id]:
                        labeled += 1
                rank += 1
                if query_id in offset2qchar and pred_pid in offset2pchar and  offset2pchar[pred_pid] == offset2qchar[query_id]:
                    # for arguana dataset
                    continue
                prediction[query_id][pred_pid] = -rank
                seen_pid.add(pred_pid)

    # use out of the box evaluation script
    evaluator = pytrec_eval.RelevanceEvaluator(
        convert_to_string_id(dev_query_positive_id), {'map_cut', 'ndcg_cut', 'recip_rank','recall'})

    eval_query_cnt = 0
    result = evaluator.evaluate(convert_to_string_id(prediction))
    
    qids_to_relevant_passageids = {}
    for qid in dev_query_positive_id:
        qid = int(qid)
        if qid in qids_to_relevant_passageids:
            pass
        else:
            qids_to_relevant_passageids[qid] = []
            for pid in dev_query_positive_id[qid]:
                if pid>0:
                    qids_to_relevant_passageids[qid].append(pid)
            
    ms_mrr = compute_metrics(qids_to_relevant_passageids, qids_to_ranked_candidate_passages)

    ndcg = 0
    Map = 0
    mrr = 0
    recall = 0
    recall_1000 = 0

    mrrs = []
    ndcgs = []

    for k in result.keys():
        eval_query_cnt += 1
        ndcg += result[k]["ndcg_cut_10"]
        Map += result[k]["map_cut_10"]
        mrr += result[k]["recip_rank"]
        recall += result[k]["recall_"+str(topN)]
        mrrs.append(float(result[k]["recip_rank"]))
        ndcgs.append(float(result[k]["ndcg_cut_10"]))

    final_ndcg = ndcg / eval_query_cnt
    final_Map = Map / eval_query_cnt
    final_mrr = mrr / eval_query_cnt
    final_recall = recall / eval_query_cnt
    hole_rate = labeled/total
    Ahole_rate = Alabeled/Atotal

    return final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, result, prediction, mrrs, ndcgs 

dev_query_embedding = []
dev_query_embedding2id = []
passage_embedding = []
passage_embedding2id = []
for i in range(8):
    try:
        with open(os.path.join(checkpoint_path, "dev_query_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb"), 'rb') as handle:
            dev_query_embedding.append(pickle.load(handle))
        with open(os.path.join(checkpoint_path, "dev_query_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb"), 'rb') as handle:
            dev_query_embedding2id.append(pickle.load(handle))
        with open(os.path.join(checkpoint_path, "passage_"+str(checkpoint)+"__emb_p__data_obj_"+str(i)+".pb"), 'rb') as handle:
            passage_embedding.append(pickle.load(handle))
        with open(os.path.join(checkpoint_path, "passage_"+str(checkpoint)+"__embid_p__data_obj_"+str(i)+".pb"), 'rb') as handle:
            passage_embedding2id.append(pickle.load(handle))
    except:
        break
if (not dev_query_embedding) or (not dev_query_embedding2id) or (not passage_embedding) or not (passage_embedding2id):
    print("No data found for checkpoint: ",checkpoint)

dev_query_embedding = np.concatenate(dev_query_embedding, axis=0)
dev_query_embedding2id = np.concatenate(dev_query_embedding2id, axis=0)
passage_embedding = np.concatenate(passage_embedding, axis=0)
passage_embedding2id = np.concatenate(passage_embedding2id, axis=0)

dim = passage_embedding.shape[1]
faiss.omp_set_num_threads(16)
cpu_index = faiss.IndexFlatIP(dim)
cpu_index.add(passage_embedding)    
dev_D, dev_I = cpu_index.search(dev_query_embedding, topN)


result = EvalDevQuery(dev_query_embedding2id, passage_embedding2id, dev_query_positive_id, dev_I, topN)
final_ndcg, eval_query_cnt, final_Map, final_mrr, final_recall, hole_rate, ms_mrr, Ahole_rate, metrics, prediction, mrrs, ndcgs  = result
print("_______________________________________")
print("DATASET: %s"%(args.dataset))
print("Results for checkpoint "+str(checkpoint))
print("NDCG@10:" + str(final_ndcg))
print("map@10:" + str(final_Map))
print("pytrec_mrr:" + str(final_mrr))
print("recall@"+str(topN)+":" + str(final_recall))
print("hole rate@10:" + str(hole_rate))
print("hole rate:" + str(Ahole_rate))
print("ms_mrr:" + str(ms_mrr))
print("_______________________________________")
