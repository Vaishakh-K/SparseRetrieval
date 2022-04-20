#!/bin/bash
#SBATCH --job-name=reforamtqry_coil_vk
#SBATCH -N 1 # Same machine
#SBATCH -t 0 # unlimited time for executing
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 32000 # memory


cd /bos/tmp16/vkeshava/coil/COIL

MODEL_DIR=/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_splade
ENCODE_QRY_OUT_DIR=${MODEL_DIR}/msmarco-query-encoding/
QUERY_DIR=${MODEL_DIR}/msmarco-query-encoding/embeddings_coil_condenser_query_reformat

echo "reformatting"
python /bos/tmp16/vkeshava/coil/COIL/retriever/format-query.py --dir $ENCODE_QRY_OUT_DIR --save_to $QUERY_DIR --as_torch



# Use the command directly. Slurm not working
# python /bos/tmp16/vkeshava/coil/COIL/retriever/format-query.py --dir /bos/tmp3/vkeshava/data/output/embeddings_coil_condenser_query --save_to /bos/tmp3/vkeshava/data/output/embeddings_coil_condenser_query_reformat --as_torch
