import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, HeNormal

"""
    Here we only use ResNet18 for PODNet on ImageNet1k, So I just implement the ResNet18
    
    :input: tensor(batch, channel, H, W)
    :return: list(3): [raw_features, features, attention]
    list[0](raw_features):  tensor (batch, 128) 
    list[1](features):      tensor (batch, 128) 
    list[2](attentions):    list(4):    list[2][0]: (batch, 16, H/2, W/2)       
                                        list[2][1]: (batch, 32, H/4, W/4)   
                                        list[2][2]: (batch, 64, H/8, W/8)
                                        list[2][3]: (batch, 128, H/16, W/16)      
"""


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, pad_mode='pad', padding=1, has_bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, pad_mode='pad', padding=0, has_bias=False)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.9,
                                  gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.9,
                                  gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.last_relu:
            out = self.relu(out)

        return out


class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, last_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.9,
                                  gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.9,
                                  gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion, eps=1e-4, momentum=0.9,
                                  gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride
        self.last_relu = last_relu

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if self.last_relu:
            out = self.relu(out)

        return out


class ResNet(nn.Cell):

    def __init__(
            self,
            block,
            layers,
            zero_init_residual=True,
            nf=16,
            last_relu=False,
            initial_kernel=3,
            **kwargs
    ):
        super(ResNet, self).__init__()

        self.last_relu = last_relu
        self.inplanes = nf
        self.conv1 = nn.Conv2d(3, nf, kernel_size=initial_kernel, stride=1, pad_mode='pad', padding=1, has_bias=False)
        self.bn1 = nn.BatchNorm2d(nf, eps=1e-4, momentum=0.9,
                                  gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, pad_mode='same')
        self.layer1 = self._make_layer(block, 1 * nf, layers[0])
        self.layer2 = self._make_layer(block, 2 * nf, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 4 * nf, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 8 * nf, layers[3], stride=2, last=True)
        # self.avgpool = ops.AdaptiveAvgPool2D((1, 1))
        self.avgpool = ops.ReduceMean(keep_dims=True)

        self.out_dim = 8 * nf * block.expansion
        print("Features dimension is {}.".format(self.out_dim))

        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), m.weight.shape, ms.float32)

        """
        Zero-initialize the last BN in each residual branch, so that the residual branch starts with zeros, 
        and each residual block behaves like an identity.
        This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        
        Instead of Pytorch's 
            if zero_init_residual:
                for m in self.cells():
                    if isinstance(m, Bottleneck):
        
        In MindSpore, this has been shown int each BatchNorm
        """

    def _make_layer(self, block, planes, blocks, stride=1, last=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-4, momentum=0.9,
                               gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            if i == blocks - 1 or last:
                layers.append(block(self.inplanes, planes, last_relu=False))
            else:
                layers.append(block(self.inplanes, planes, last_relu=self.last_relu))

        return nn.SequentialCell(*layers)

    @property
    def last_block(self):
        return self.layer4

    @property
    def last_conv(self):
        return self.layer4[-1].conv2

    def construct(self, x):
        # print(x.shape)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_1 = self.layer1(x)
        x_2 = self.layer2(self.end_relu(x_1))
        x_3 = self.layer3(self.end_relu(x_2))
        x_4 = self.layer4(self.end_relu(x_3))

        raw_features = self.end_features(x_4)
        relu = ops.ReLU()
        features = self.end_features(relu(x_4))

        attention = [x_1, x_2, x_3, x_4]

        return [raw_features, features, attention]
        # return {
        #     "raw_features": raw_features,
        #     "features": features,
        #     "attention": [x_1, x_2, x_3, x_4]
        # }

    def end_features(self, x):
        x = self.avgpool(x, (2, 3))
        x = x.view(x.shape[0], -1)  # size now is (batch, channel)
        return x

    def end_relu(self, x):
        if hasattr(self, "last_relu") and self.last_relu:
            relu = ops.ReLU()
            return relu(x)
        return x


def resnet18(**kwargs):
    """
    Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


"""
Unit Test
    5.24 the same shape result with Pytorch
    The following is the test code
    Result: Fine to work on Ascend910, dict return is fine on PYNATIVE_MODE, but not fine on GRAPH_MODE
"""
# from mindspore import Tensor, context
# import numpy as np

#
# context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend', device_id=3)
# context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
# x = resnet18()
# y = Tensor(np.random.randn(3, 3, 224, 224).astype(np.float32))
# y = x(y)
# print(y['raw_features'].shape)  # (3, 128)
# print(y['features'].shape)  # (3, 128)
# print(y['attention'][0].shape)  # (3, 16, 112, 112)
# print(y['attention'][1].shape)  # (3, 32, 56, 56)
# print(y['attention'][2].shape)  # (3, 64, 28, 28)

# print(y[0].shape)  # (3, 128)
# print(y[1].shape)  # (3, 128)
# print(y[2][0].shape)  # (3, 16, 112, 112)
# print(y[2][1].shape)  # (3, 32, 56, 56)
# print(y[2][2].shape)  # (3, 64, 28, 28)
# print(y[2][3].shape)  # (3, 128, 14, 14)
