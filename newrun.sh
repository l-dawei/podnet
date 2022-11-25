#!/bin/bash
# applicable to Ascend

echo "=============================================================================================================="
echo "Please run the script as: "
echo "bash run.sh"
echo "For example: bash run.sh /path/dataset 8"
echo "It is better to use the absolute path."
echo "=============================================================================================================="
set -e
#DATA_PATH=$1
#export DATA_PATH=${DATA_PATH}
ulimit -u unlimited
ulimit -n 655260

#RANK_SIZE=$4

EXEC_PATH=$(pwd)

#test_dist_8pcs()
#{
#    export RANK_TABLE_FILE=${EXEC_PATH}/rank_table_8pcs.json
#    export RANK_SIZE=8
#}

#test_dist_2pcs()
#{
#
#}
#test_dist_${RANK_SIZE}pcs

#export RANK_TABLE_FILE=${EXEC_PATH}/hccl_4p_0123_127.0.0.1.json
export RANK_TABLE_FILE=${EXEC_PATH}/hccl_8p_01234567_127.0.0.1.json
export RANK_SIZE=8

for((i=1;i<${RANK_SIZE};i++))
do
    rm -rf device$i
    mkdir device$i
#    cp ./__main__.py ./resnet.py ./device$i
#    cd ./device$i
    export DEVICE_ID=$i
    export RANK_ID=$i
    echo "start training for device $i"
    env > ./device$i/env$i.log
#    pytest -s -v ./resnet50_distributed_training.py > train.log$i 2>&1 &
    python3 -minclearn > ./device$i/train.log$i 2>&1 &
#    cd ../
done

rm -rf device0
mkdir device0
#cp ./resnet50_distributed_training.py ./resnet.py ./device0
#cd ./device0
export DEVICE_ID=0
export RANK_ID=0
echo "start training for device 0"
env > ./device0/env0.log
#pytest -s -v ./resnet50_distributed_training.py > train.log0 2>&1
python3 -minclearn > ./device0/train.log0 2>&1

if [ $? -eq 0 ];then
    echo "training success"
else
    echo "training failed"
    exit 2
fi
#cd ../