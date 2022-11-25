import collections
from mindspore import Tensor, Parameter, context
import mindspore as ms
import copy
import numpy as np
from mindspore.communication import init, get_rank, get_group_size

# _metrics = collections.defaultdict(float)
# print(_metrics)
# aa = Parameter(Tensor(1.0, ms.float32))
# context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=1)
# aa = Tensor(1.0, ms.float32)
# print(aa + 1.1)
# aaa = copy.deepcopy(aa)
# print(aaa)

# context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

# x = Tensor(np.random.randn(128, 3, 256, 256).astype(np.float32))
# y = Tensor(np.random.randn(128, 3, 256, 256).astype(np.float32))
# xx = [x, y]
# z = Tensor(np.random.randn(128, 3, 256, 256).astype(np.float32))
# m = Tensor(np.random.randn(128, 3, 128, 256).astype(np.float32))
# vv = [z, m]
# for mm in vv:
#     xx.append(mm)
# # xx.append(z)
# print(len(xx))
init()
# device_num = get_group_size()
# rank = get_rank()
# print("rank_id is {}, device_num is {}".format(rank, device_num))
ms.reset_auto_parallel_context()
# 下述的并行配置用户只需要配置其中一种模式
# 数据并行模式
ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL)