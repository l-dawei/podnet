from mindspore.communication import init, get_rank, get_group_size
from mindspore import context

import os

os.environ['RANK_ID'] = '2'

init()
device_num = get_group_size()
rank = get_rank()
print("rank_id is {}, device_num is {}".format(rank, device_num))
context.reset_auto_parallel_context()
# 下述的并行配置用户只需要配置其中一种模式
# 数据并行模式
context.set_auto_parallel_context(parallel_mode=context.ParallelMode.DATA_PARALLEL)
