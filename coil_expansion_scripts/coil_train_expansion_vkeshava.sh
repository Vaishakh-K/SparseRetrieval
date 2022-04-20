#!/bin/bash
#SBATCH --job-name=train-coil_condenser
#SBATCH --output=/home/adityasv/COILv2/SparseRetrieval/train_logs/%x-%j.out
#SBATCH -N 1 # Same machine
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 32000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-user=adityasv@andrew.cmu.edu

cd /home/adityasv/COILv2/SparseRetrieval/COIL

TRAIN=/bos/tmp16/vkeshava/coil/data/psg-train/ # WITHOUT DOC2QUERY
OUTPUT=/bos/tmp15/adityasv/COILv2-vkeshava/debug
# OUTPUT=/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_cls_flops_q0.0008_d0.0006_expansion_weight1_tokens_weight1_expansion-norm_cls_expansion_weight_not_learned_init_distilbert_base

# TRAIN=/bos/tmp3/vkeshava/data/psg-train-d2q # WITH DOC2QUERY
# OUTPUT=/bos/tmp15/adityasv/COILv2-vkeshava/model_coil_with_expansion_splade_flops_d2q

export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1 

MASTER_PORT=12345

echo $OUTPUT

# python run_marco.py \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=$MASTER_PORT run_marco.py \
  --output_dir ${OUTPUT} \
  --model_name_or_path distilbert-base-uncased \
  --do_train \
  --save_steps 10000 \
  --train_dir ${TRAIN} \
  --q_max_len 16 \
  --p_max_len 128 \
  --fp16 \
  --per_device_train_batch_size 4 \
  --train_group_size 8 \
  --gradient_accumulation_steps 2 \
  --cls_dim 768 \
  --token_dim 32 \
  --warmup_ratio 0.1 \
  --learning_rate 5e-6 \
  --num_train_epochs 5 \
  --overwrite_output_dir \
  --dataloader_num_workers 16 \
  --no_sep \
  --pooling max \
  --no_cls \
  --do_expansion True \
  --expansion_weight 1 \
  --tokens_weight 1 \
  --lambda_q 0.8 \
  --lambda_d 0.6 \
  --expansion_normalization cls \
  --splade_separate_loss False \
  --expansion_weight_learned False 


