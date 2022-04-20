#!/bin/bash
#SBATCH --job-name=index_shard_coil_condenser
#SBATCH -N 1 # Same machine
#SBATCH -t 0 # unlimited time for executing
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 64000 # memory

cd /home/adityasv/COILv2/SparseRetrieval/COIL

MODEL_DIR=$1

ENCODE_OUT_DIR=${MODEL_DIR}/msmarco-passage-encoding/
INDEX_DIR=${MODEL_DIR}/msmarco-passage-encoding-index/

mkdir -p $INDEX_DIR

for i in $(seq -f "%02g" 0 0)
do  
 echo "$i starting"
#  srun --ntasks=1 -c4 --mem=32G -t0 -p cpu --nodelist=boston-2-38 python retriever/sharding.py \
python retriever/sharding.py \
 --n_shards 20 \
 --shard_id $i \
 --dir $ENCODE_OUT_DIR \
 --save_to $INDEX_DIR \
 --use_torch
 echo "$i done"
done 
