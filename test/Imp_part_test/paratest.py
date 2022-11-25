import mindspore.nn as nn
import mindspore as ms
# import logging
#
# logger = logging.getLogger(__name__)
from mindspore.common.initializer import initializer, HeNormal

"""
Test of para on lenet 5 
"""


class LeNet5(nn.Cell):
    """
    Lenet
    """

    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        # 定义所需要的运算
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120)
        self.fc2 = nn.Dense(120, 84)
        self.fc3 = nn.Dense(84, num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

        # for m in self.cells():
        #     # print(m)
        #     # logger.info(m)
        #     if isinstance(m, nn.Dense):
        #         print("Dense")
        #         print(m.weight)
        #         print(m.weight.shape)
        #         print(type(m.weight))
        #         print(m.weight.asnumpy())
        #         print('---------------')
        #         m.weight = initializer(HeNormal(), m.weight.shape, ms.float32)
        #         print(m.weight.asnumpy())
        #
        #         # print(initializer())
        #         # logger.info("Conv2d")
        #         # initializer(HeNormal(), m.weight, ms.float32)
        #     elif isinstance(m, nn.MaxPool2d):
        #         print("Max2d")
        #         # nn.init.constant_(m.weight, 1)
        #         # nn.init.constant_(m.bias, 0)
        #
        #     elif isinstance(m, nn.Dense):
        #         print("Dense")

    def construct(self, x):
        # 使用定义好的运算构建前向网络
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


x = LeNet5()
x1 = x.trainable_params()
# print(type(x1))  # list
for i in x1:
    print(i)
    # Parameter(name=conv1.weight, shape=(6, 1, 5, 5), dtype=Float32, requires_grad=True)
    # Parameter(name=conv2.weight, shape=(16, 6, 5, 5), dtype=Float32, requires_grad=True)
    # Parameter(name=fc1.weight, shape=(120, 400), dtype=Float32, requires_grad=True)
    # Parameter(name=fc1.bias, shape=(120,), dtype=Float32, requires_grad=True)
    # Parameter(name=fc2.weight, shape=(84, 120), dtype=Float32, requires_grad=True)
    # Parameter(name=fc2.bias, shape=(84,), dtype=Float32, requires_grad=True)
    # Parameter(name=fc3.weight, shape=(10, 84), dtype=Float32, requires_grad=True)
    # Parameter(name=fc3.bias, shape=(10,), dtype=Float32, requires_grad=True)
