import os
from mindspore import Tensor
import numpy as np

# x = os.getcwd()
# print(x)
# print(os.path.dirname(x))

z = Tensor(np.random.randn(128, 3, 256, 256).astype(np.float32))
print(type(z.shape[1] / 10))
