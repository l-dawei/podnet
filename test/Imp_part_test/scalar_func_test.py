from mindspore import nn
import mindspore as ms
from mindspore import Tensor
import numpy as np

initial_value = 1.
x = ms.Parameter(ms.Tensor(initial_value))

similarities = np.random.randn(128, 10).astype(np.float32)
similarities = Tensor(similarities)

y = similarities * x
print(y)
# print(type(y))
