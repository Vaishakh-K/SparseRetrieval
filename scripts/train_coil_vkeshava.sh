#!/bin/bash
#SBATCH --job-name=train-coil_condenser
#SBATCH --output=./train_logs/%x-%j.out
#SBATCH -N 1 # Same machine
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 32000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p gpu
#SBATCH --gres=gpu:4

cd /home/adityasv/COILv2/SparseRetrieval/COIL

TRAIN=/bos/tmp16/vkeshava/coil/data/psg-train/ # WITHOUT DOC2QUERY
OUTPUT=/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion

# TRAIN=/bos/tmp3/vkeshava/data/psg-train-d2q # WITH DOC2QUERY
# OUTPUT=/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_condenser-d2q

export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1 

python -m torch.distributed.launch --nproc_per_node=4 run_marco.py \
  --output_dir ${OUTPUT} \
  --model_name_or_path bert-base-uncased \
  --do_train \
  --save_steps 4000 \
  --train_dir ${TRAIN} \
  --q_max_len 16 \
  --p_max_len 128 \
  --fp16 \
  --per_device_train_batch_size 8 \
  --train_group_size 8 \
  --cls_dim 768 \
  --token_dim 32 \
  --warmup_ratio 0.1 \
  --learning_rate 5e-6 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --no_sep \
  --pooling max \
  --no_cls
