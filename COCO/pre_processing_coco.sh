
name=corpus # the path to save the processed data
JSON_SAVE_DIR=${name}
mkdir -p $JSON_SAVE_DIR
i=0
for s in trec-covid bioasq_correct nfcorpus nq hotpotqa fiqa robust04 trec-news arguana webis-touche2020 cqadupstack climate-fever scidocs scifact signal1m msmarco quora dbpedia-entity 
do
 file_s="./BEIR/${s}/corpus.jsonl"
 echo "Dataset ${i}: ${s}, ${name}"
 python helper/create_train_co_short.py \
  --tokenizer bert-base-uncased \
  --file ${file_s} \
  --save_to $JSON_SAVE_DIR/corpus_${i}.json \

  i=$((i+1))
done
