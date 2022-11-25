import mindspore.nn as nn
import numpy as np
from mindspore import Tensor

"""
A test on why i can't get my dict return in construct func
"""


class te(nn.Cell):
    def __init__(self):
        super(te, self).__init__()
        self.cnn1 = nn.Conv2d(3, 6, stride=1, kernel_size=3, has_bias=False, pad_mode='same')
        self.x1 = Tensor(np.random.randn(128, 3, 224, 224).astype(np.float32))
        self.x2 = Tensor(np.random.randn(128, 3, 224, 224).astype(np.float32))

    def construct(self, x):
        # print(x.shape)
        x = self.cnn1(x)

        x3 = [self.x1, self.x2]
        return {"aa": x3, "bb": x}


x = Tensor(np.random.randn(128, 3, 224, 224).astype(np.float32))
y = te()
x = y(x)
print(x)

import mindspore.ops as ops
import numpy as np
from mindspore import Tensor, context

# context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend', device_id=2)

op = ops.ReduceMean(keep_dims=True)
x = Tensor(np.random.randn(128, 3, 224, 224).astype(np.float32))
y = op(x, (2, 3))  # y is (128, 3, 1, 1)

y = y.view(x.shape[0], -1)  # y is (128, 3), this proves that in mindspore -1 is yes in view
print(y.shape)
