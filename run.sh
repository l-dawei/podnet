#!/bin/bash

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh"
echo "For example: bash run.sh"
echo "It is better to use the absolute path."
echo "=============================================================================================================="

#DATA_PATH=$1
#export DATA_PATH=${DATA_PATH}
#RANK_SIZE=$2
export DEVICE_NUM=8
export RANK_SIZE=8
#export HCCL_CONNECT_TIMEOUT=6000

EXEC_PATH=$(pwd)

#python -minclearn

for((i=0;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
#    cp ./resnet50_distributed_training.py ./resnet.py ./device$i
#    cd ./device$i
    export RANK_TABLE_FILE=${EXEC_PATH}/hccl_2p_01_127.0.0.1.json
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > ./device$i/env$i.log
    python3 -minclearn > ./device$i/train.log$i 2>&1
done
