cd ../data/raw_data/

# download MSMARCO passage data
wget https://msmarco.blob.core.windows.net/msmarcoranking/collectionandqueries.tar.gz
tar -zxvf collectionandqueries.tar.gz
rm collectionandqueries.tar.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/msmarco-passagetest2019-top1000.tsv.gz
gunzip msmarco-passagetest2019-top1000.tsv.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/top1000.dev.tar.gz
tar -zxvf top1000.dev.tar.gz
rm top1000.dev.tar.gz

wget https://msmarco.blob.core.windows.net/msmarcoranking/triples.train.small.tar.gz
tar -zxvf triples.train.small.tar.gz
rm triples.train.small.tar.gz

