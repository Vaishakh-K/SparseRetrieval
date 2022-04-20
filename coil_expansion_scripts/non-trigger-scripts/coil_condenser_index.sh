#!/bin/bash
#SBATCH --job-name=index_shard_coil_condenser
#SBATCH -N 1 # Same machine
#SBATCH -t 0 # unlimited time for executing
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 64000 # memory

cd /bos/tmp16/vkeshava/coil/COIL

MODEL_DIR=/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_splade

ENCODE_OUT_DIR=${MODEL_DIR}/msmarco-passage-encoding/
INDEX_DIR=${MODEL_DIR}/msmarco-passage-encoding-index/

mkdir -p $INDEX_DIR

for i in $(seq 0 99)  
do  
 echo "$i starting"
 python retriever/sharding.py \
 --n_shards 100 \
 --shard_id $i \
 --dir $ENCODE_OUT_DIR \
 --save_to $INDEX_DIR \
 --use_torch
 echo "$i done"
done
