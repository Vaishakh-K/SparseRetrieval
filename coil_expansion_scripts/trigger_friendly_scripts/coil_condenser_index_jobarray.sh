#!/bin/bash
#SBATCH --job-name=index_shard_coil_condenser
#SBATCH -N 1 # Same machine
#SBATCH -t 0 # unlimited time for executing
#SBATCH -n 8 # Nr of cores
#SBATCH --mem 64G # memory
#SBATCH -p cpu

cd /home/adityasv/COILv2/COIL-vkeshava/COIL

MODEL_DIR=$1
N_SHARDS=100

ENCODE_OUT_DIR=${MODEL_DIR}/msmarco-passage-encoding/
INDEX_DIR=${MODEL_DIR}/msmarco-passage-encoding-index/

mkdir -p $INDEX_DIR

taskid=$SLURM_ARRAY_TASK_ID
printf -v i "%02d" $taskid

echo "n shards: $N_SHARDS"
echo "$i starting"

python retriever/sharding.py \
 --n_shards $N_SHARDS \
 --shard_id $i \
 --dir $ENCODE_OUT_DIR \
 --save_to $INDEX_DIR \
 --use_torch

echo "$i done"
