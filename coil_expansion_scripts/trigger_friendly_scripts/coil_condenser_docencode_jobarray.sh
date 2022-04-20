#!/bin/bash
#SBATCH --job-name=encode-doc-coil_condenser
#SBATCH -N 1 # Same machine
#SBATCH --mem 32000 # memory
#SBATCH -t 0 # unlimited time for executing
#SBATCH -p gpu
#SBATCH --gres=gpu:1

export NCCL_SOCKET_IFNAME=eno1
export NCCL_IB_DISABLE=1 


cd /home/adityasv/COILv2/SparseRetrieval/COIL/

MODEL_DIR=$1
ENCODE_OUT_DIR=${MODEL_DIR}/msmarco-passage-encoding/
CKPT_DIR=$MODEL_DIR
CORPUS=/bos/tmp3/vkeshava/data/corpus

taskid=$SLURM_ARRAY_TASK_ID
printf -v i "%02d" $taskid
echo "processing split:" $i

mkdir -p $ENCODE_OUT_DIR
mkdir -p ${ENCODE_OUT_DIR}/split${i}

python run_marco.py \
    --output_dir $ENCODE_OUT_DIR \
    --model_name_or_path $CKPT_DIR \
    --tokenizer_name bert-base-uncased  \
    --token_dim 32 \
    --cls_dim 768 \
    --do_encode \
    --no_sep \
    --no_cls \
    --do_expansion True \
    --p_max_len 128 \
    --pooling max \
    --fp16 \
    --per_device_eval_batch_size 128 \
    --dataloader_num_workers 12 \
    --encode_in_path ${CORPUS}/split${i} \
    --encoded_save_path ${ENCODE_OUT_DIR}/split${i} \
    --expansion_weight 0.1

echo "done"