#!/bin/bash
#SBATCH --job-name=encode-doc-coil_condenser
#SBATCH -N 1 # Same machine
#SBATCH -n 16 # Nr of cores
#SBATCH --mem 32000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p gpu
#SBATCH --gres=gpu:1

export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1 


cd /home/adityasv/COILv2/SparseRetrieval/COIL/


MODEL_DIR=/bos/tmp15/adityasv/COILv2/attention-recovery-32dim
ENCODE_QRY_OUT_DIR=$MODEL_DIR/msmarco-query-encoding/
CKPT_DIR=$MODEL_DIR
QUERY=/bos/tmp16/vkeshava/coil/data/queries.dev.small.json

mkdir -p $ENCODE_QRY_OUT_DIR

python run_marco.py \
  --output_dir $ENCODE_QRY_OUT_DIR \
  --model_name_or_path $CKPT_DIR \
  --tokenizer_name Luyu/co-condenser-marco \
  --token_dim 32 \
  --cls_dim 768 \
  --do_encode \
  --p_max_len 16 \
  --fp16 \
  --no_sep \
  --pooling max \
  --per_device_eval_batch_size 128 \
  --dataloader_num_workers 12 \
  --encode_in_path ${QUERY} \
  --encoded_save_path $ENCODE_QRY_OUT_DIR
