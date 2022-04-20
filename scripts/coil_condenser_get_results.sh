#!/bin/bash
#SBATCH --job-name=get_results_coil_condenser
#SBATCH -N 1 # Same machine
#SBATCH -t 0 # unlimited time for executing
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 32000 # memory

cd /home/adityasv/COILv2/COIL-vkeshava/COIL

MODEL_DIR=/bos/tmp15/adityasv/COILv2/attention-recovery-32dim
SCORE_DIR=${MODEL_DIR}/coilcondenserscores
SCORE_INTERMEDIATE_DIR=${MODEL_DIR}/coilcondenserscores/intermediate
QUERY_DIR=${MODEL_DIR}/msmarco-query-encoding/embeddings_coil_condenser_query_reformat

mkdir -p $SCORE_INTERMEDIATE_DIR

python retriever/merger.py \
  --score_dir $SCORE_INTERMEDIATE_DIR \
  --query_lookup  $QUERY_DIR/cls_ex_ids.pt \
  --depth 1000 \
  --save_ranking_to $MODEL_DIR/rank.txt

python data_helpers/msmarco-passage/score_to_marco.py \
  --score_file $MODEL_DIR/rank.txt

# Get the metrics
python /bos/tmp16/vkeshava/tools/anserini/tools/scripts/msmarco/msmarco_passage_eval.py \
        /bos/tmp16/vkeshava/tools/anserini/tools/topics-and-qrels/qrels.msmarco-passage.dev-subset.txt \
        $MODEL_DIR/rank.txt.marco
