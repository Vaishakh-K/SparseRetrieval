#!/bin/bash
#SBATCH --job-name=retrieve_coil_shard
#SBATCH -N 1 # Same machine
#SBATCH -t 0 # unlimited time for executing
#SBATCH -n 4 # Nr of cores
#SBATCH --mem 32000 # memory
#SBATCH -p cpu

cd /home/adityasv/COILv2/SparseRetrieval/COIL

MODEL_DIR=$1
QUERY_DIR=${MODEL_DIR}/msmarco-query-encoding/embeddings_coil_condenser_query_reformat
INDEX_DIR=${MODEL_DIR}/msmarco-passage-encoding-index
SCORE_DIR=${MODEL_DIR}/coilcondenserscores

mkdir -p $SCORE_DIR
mkdir -p $SCORE_DIR/intermediate

taskid=$SLURM_ARRAY_TASK_ID
printf -v i "%02d" $taskid

echo "split $i"
python retriever/retriever-fast.py \
--query $QUERY_DIR \
--doc_shard $INDEX_DIR/shard_${i} \
--top 1000 \
--batch_size 512 \
--save_to ${SCORE_DIR}/intermediate/shard_${i}.pt
