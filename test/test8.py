from mindspore import Tensor
import numpy as np
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore
from PIL import Image
import mindspore.dataset as ds
from mindspore.common.initializer import initializer, HeUniform

# path1 = 'D:\\file\Dataset\\flower\\flower_photos\daisy\\5547758_eea9edfd54_n.jpg'

# aaa = ds.RandomSampler()
"""
Basic ops tests
"""
# x = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]]), mindspore.float32)
# y = Tensor(np.array([[2.0, 4.0], [2.0, 3.0]]), mindspore.float32)
# z = Tensor(np.random.randn(3, 4, 6), mindspore.float32)
# xl = [z, z, z]
# concat_op = ops.Concat()
# add_op = ops.Add()
# pow_op = ops.Pow()
# re = add_op(x, y)
# re = pow_op(x, 2)

# x1 = Tensor(1, mindspore.float32)
# le_op = ops.LessEqual()
# error_mask = le_op(x, y)
# error_mask = (error_mask == False).astype(mindspore.float32)
# error_mask = error_mask.asnumpy().astype(float)
# print(error_mask)
# re = concat_op([i for i in xl])
# print(re.shape)
"""
list test
"""
# increments = [50]
# increment = 1
# increments.extend([increment for _ in range(int(50))])
# x = [1, 3, 5, 7]
# print(sum(x))
# x = x[:1]
# print(increments)
# _current_task = 0
# min_class = sum(increments[:_current_task])
# print(min_class)
# max_class = sum(increments[:_current_task + 1])
# print(max_class)

"""
Basic test
"""
# x1 = ms.Parameter(ms.Tensor(np.zeros((2, 3)), ms.float32))
# x2 = ms.Parameter(ms.Tensor(np.zeros((2, 3)), ms.float32))
# concat_op = ops.Concat()
# x3 = concat_op((x1, x2))
# print(type(x3))  # tensor
# y = initializer(HeUniform(nonlinearity='linear'), [1, 2, 3], mindspore.float32)
# print(y)
# print(x.shape)
# print(x.dtype)

# x = ms.Tensor(np.random.randint(low=0, high=10, size=(128,), dtype=np.int32))
# print(x.dtype)
# x = x.astype(ms.int64)
# print(x.dtype)

np.random.seed(1)
# similarities = np.random.randn(128, 10).astype(np.float32)
# targets = np.random.randint(low=0, high=5, size=(128,), dtype=np.int32)
#
# similarities = Tensor(similarities)
# targets = Tensor(targets)
#
# similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability
# print(similarities.shape)  # (128, 10)
# print(len(similarities))
# x = Tensor(1.2)
# print(x > 1) # True
# if x > 1:
#     print("ssa")
from mindspore import Parameter

x = {
    'params': Parameter(), 'lr': 0.1
}

"""
has attr test

fine with graph mode in Ascend
"""

# from mindspore import context
#
# context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=0)
#
#
# class LeNet5(nn.Cell):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.conv1 = nn.Conv2d(10, 6, 5, pad_mode='valid')
#         self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
#         self.fc1 = nn.Dense(16 * 5 * 5, 120)
#         self.fc2 = nn.Dense(120, 84)
#         self.fc3 = nn.Dense(84, 10)
#         self.relu = nn.ReLU()
#         self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.flatten = nn.Flatten()
#
#     def construct(self, x):
#         x = 1
#
#         return x
#
#     def sayhello(self):
#         if hasattr(self, "conv1"):
#             print("has conv1")
#         print('hello')

# a = LeNet5()
# a.sayhello()
# # if hasattr(a, "sayhello"):
# #     print("wow")
# # print('sssss')
# a = LeNet5()
# print(a.training) # False
# a.set_train(mode=True)
# print(a.training) # True

"""
distillation test
"""
# list_attentions_a = np.random.randn(128, 3, 64, 64).astype(np.float32)
# list_attentions_b = np.random.randn(128, 3, 64, 64).astype(np.float32)
# list_attentions_a = [Tensor(list_attentions_a), Tensor(list_attentions_a), Tensor(list_attentions_a),
#                      Tensor(list_attentions_a)]
# list_attentions_b = [Tensor(list_attentions_b), Tensor(list_attentions_b), Tensor(list_attentions_b),
#                      Tensor(list_attentions_b)]
# for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
#     pow_op = ops.Pow()
#     a = pow_op(a, 2)
#     b = pow_op(b, 2)
#
#     concat_op = ops.Concat(axis=-1)
#
#     a_h = a.sum(axis=3).view(a.shape[0], -1)
#     print(a_h.shape)  # (128, 192)
#     b_h = b.sum(axis=3).view(b.shape[0], -1)
#     print(b_h.shape)  # (128, 192)
#     a_w = a.sum(axis=2).view(a.shape[0], -1)
#     print(a_w.shape)  # (128, 192)
#     b_w = b.sum(axis=2).view(b.shape[0], -1)
#     print(b_w.shape)  # (128, 192)
#     a = concat_op([a_h, a_w])
#     print(a.shape)  # (128, 384)
#     b = concat_op([b_h, b_w])
#     print(b.shape)  # (128, 384)
#
#     l2_normalize = ops.L2Normalize(dim=1)
#     a = l2_normalize(a)
#     b = l2_normalize(b)
#
#     mean = ops.ReduceMean(keep_dims=False)
#
#     if i >= 10:
#         break
