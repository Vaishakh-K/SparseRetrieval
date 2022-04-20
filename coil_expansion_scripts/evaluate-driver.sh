#!/bin/bash
#SBATCH --job-name=encode-doc-coil_condenser
#SBATCH -N 1 # Same machine
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 64000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adityasv@andrew.cmu.edu

checkpoint_path=$1

bash trigger_friendly_scripts/coil_condenser_docencode.sh $1 && \
bash trigger_friendly_scripts/coil_condenser_qryencode.sh $1 && \
bash trigger_friendly_scripts/coil_condenser_reformatqry.sh $1 && \
bash trigger_friendly_scripts/coil_condenser_index.sh $1 && \
bash trigger_friendly_scripts/coil_condenser_retrieve_shard.sh $1 && \
bash trigger_friendly_scripts/coil_condenser_get_results.sh $1
