import mindspore
from mindspore import Tensor
from mindspore import dtype as mstype
from mindspore.common.initializer import One, Normal
import mindspore.ops as ops

a = Tensor(shape=(4, 1), dtype=mstype.float32, init=Normal())
# b = Tensor(ops.StandardNormal(3, 4))
shape = (1, 4)
stdnormal = ops.StandardNormal(seed=2)
b = stdnormal(shape)
print(type(a))
# print(a)

print(type(b))
# print(a + b)
mul = ops.Mul()
c = mul(a, 0)
print(c.shape)
relu = mindspore.ops.ReLU()
y = relu(c)
print(c.shape)