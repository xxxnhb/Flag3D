#!/usr/bin/env bash

set -x

CONFIG=$1
GPUS=$2
PORT=$3
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py \
    $CONFIG \
    --launcher pytorch ${@:4}
