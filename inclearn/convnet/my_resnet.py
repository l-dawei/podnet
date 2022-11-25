"""
MindSpore port of the resnet used for CIFAR100 by iCaRL.

https://github.com/srebuffi/iCaRL/blob/master/iCaRL-TheanoLasagne/utils_cifar100.py

:input: tensor(batch, channel, H, W) eg:(3, 3, 224, 224)
:return: list(3): [raw_features, features, attentions]
    list[0](raw_features):  tensor (batch, 64) eg:(3, 64)
    list[1](features):      tensor (batch, 64) eg:(3, 64)
    list[2](attentions):    list(3):    list[2][0]: (batch, 16, H, W)       eg:(3, 16, 224, 224)
                                        list[2][1]: (batch, 32, H/2, W/2)   eg:(3, 32, 112, 112)
                                        list[2][2]: (batch, 64, H/4, W/4)   eg:(3, 32, 64, 64)
"""
import logging
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
# from inclearn.lib import pooling
from mindspore.common.initializer import initializer, HeNormal

logger = logging.getLogger(__name__)


# shape is right
class DownsampleStride(nn.Cell):
    def __init__(self, n=2):
        super(DownsampleStride, self).__init__()
        self._n = n

    def construct(self, x):
        return x[..., ::2, ::2]


class DownsampleConv(nn.Cell):
    def __init__(self, inplanes, planes):
        super().__init__()

        self.conv = nn.SequentialCell(
            nn.Conv2d(inplanes, planes, stride=2, kernel_size=1, has_bias=False, pad_mode='same'),
            nn.BatchNorm2d(planes, eps=1e-4, momentum=0.9,
                           gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1),
        )

    def construct(self, x):
        return self.conv(x)


# shape is right
class ResidualBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, increase_dim=False, last_relu=False, downsampling="stride"):
        super(ResidualBlock, self).__init__()

        self.increase_dim = increase_dim

        if increase_dim:
            first_stride = 2
            planes = inplanes * 2
        else:
            first_stride = 1
            planes = inplanes

        self.conv_a = nn.Conv2d(inplanes, planes, kernel_size=3, stride=first_stride, pad_mode='pad', padding=1,
                                has_bias=False)
        self.bn_a = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.9,
                                   gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                                has_bias=False)
        self.bn_b = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.9,
                                   gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)

        if increase_dim:
            if downsampling == "stride":
                self.downsampler = DownsampleStride()
                self._need_pad = True
            else:
                self.downsampler = DownsampleConv(inplanes, planes)
                self._need_pad = False
        self.last_relu = last_relu

    """
    here maybe some problem about mindspore.ops.Concat because of the precision
    """

    @staticmethod
    def pad(x):
        mul = ops.Mul()
        concat = ops.Concat(axis=1)
        return concat((x, mul(x, 0)))

    def construct(self, x):
        y = self.conv_a(x)
        y = self.bn_a(y)
        relu = ops.ReLU()
        y = relu(y)

        y = self.conv_b(y)
        y = self.bn_b(y)

        if self.increase_dim:
            x = self.downsampler(x)
            if self._need_pad:
                x = self.pad(x)

        y = x + y

        if self.last_relu:
            relu = ops.ReLU()
            y = relu(y)

        return y


class PreActResidualBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, increase_dim=False, last_relu=False):
        super().__init__()

        self.increase_dim = increase_dim

        if increase_dim:
            first_stride = 2
            planes = inplanes * 2
        else:
            first_stride = 1
            planes = inplanes

        self.bn_a = nn.BatchNorm2d(inplanes, eps=1e-4, momentum=0.9,
                                   gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)
        self.conv_a = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=first_stride, pad_mode='pad', padding=1,
            has_bias=False)

        self.bn_b = nn.BatchNorm2d(planes, eps=1e-4, momentum=0.9,
                                   gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

        self.conv_b = nn.Conv2d(planes, planes, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                                has_bias=False)

        if increase_dim:
            self.downsample = DownsampleStride()
            mul = ops.Mul()
            concat = ops.Concat(axis=1)
            self.pad = lambda x: concat((x, mul(x, 0)))
        self.last_relu = last_relu

    def construct(self, x):
        y = self.bn_a(x)
        relu = ops.ReLU()
        y = relu(y)
        y = self.conv_a(x)

        y = self.bn_b(y)
        relu = ops.ReLU()
        y = relu(y)
        y = self.conv_b(y)

        if self.increase_dim:
            x = self.downsample(x)
            x = self.pad(x)

        y = x + y

        if self.last_relu:
            relu = ops.ReLU()
            y = relu(y)
        return y


class Stage(nn.Cell):
    def __init__(self, blocks, block_relu=False):
        super().__init__()

        self.blocks = nn.CellList(blocks)
        self.block_relu = block_relu

    def construct(self, x):
        intermediary_features = []

        for b in self.blocks:
            x = b(x)
            intermediary_features.append(x)

            if self.block_relu:
                relu = ops.ReLU()
                x = relu(x)

        return intermediary_features, x


class CifarResNet(nn.Cell):
    """
    ResNet optimized for the Cifar Dataset, as specified in
    https://arxiv.org/abs/1512.03385.pdf
    """

    def __init__(
            self,
            n=5,
            nf=16,
            channels=3,
            preact=False,
            zero_residual=True,
            pooling_config={"type": "avg"},
            downsampling="stride",
            final_layer=False,
            all_attentions=False,
            last_relu=False,
            **kwargs
    ):
        """
        Constructor
        Args:
            depth: number of layers.
            num_classes: number of classes
            base_width: base width
        """
        if kwargs:
            raise ValueError("Unused kwargs: {}.".format(kwargs))

        self.all_attentions = all_attentions
        logger.info("Downsampling type {}".format(downsampling))
        self._downsampling_type = downsampling
        self.last_relu = last_relu

        Block = ResidualBlock if not preact else PreActResidualBlock

        super(CifarResNet, self).__init__()

        self.conv_1_3x3 = nn.Conv2d(channels, nf, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                                    has_bias=False)
        self.bn_1 = nn.BatchNorm2d(nf, eps=1e-4, momentum=0.9,
                                   gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)

        self.stage_1 = self._make_layer(Block, nf, increase_dim=False, n=n)
        self.stage_2 = self._make_layer(Block, nf, increase_dim=True, n=(n - 1))
        self.stage_3 = self._make_layer(Block, 2 * nf, increase_dim=True, n=(n - 2))
        self.stage_4 = Block(
            4 * nf, increase_dim=False, last_relu=False, downsampling=self._downsampling_type
        )

        if pooling_config["type"] == "avg":
            # self.pool = ops.AdaptiveAvgPool2D((1, 1))  because this not support Ascend
            self.pool = ops.ReduceMean(keep_dims=True)
        # elif pooling_config["type"] == "weldon":
        #     self.pool = pooling.WeldonPool2d(**pooling_config)
        else:
            raise ValueError("Unknown pooling type {}.".format(pooling_config["type"]))

        self.out_dim = 4 * nf
        if final_layer in (True, "conv"):
            self.final_layer = nn.Conv2d(self.out_dim, self.out_dim, kernel_size=1, has_bias=False)
        elif isinstance(final_layer, dict):
            if final_layer["type"] == "one_layer":
                self.final_layer = nn.SequentialCell(
                    nn.BatchNorm1d(self.out_dim), nn.ReLU(),
                    nn.Dense(self.out_dim, int(self.out_dim * final_layer["reduction_factor"]))
                )
                self.out_dim = int(self.out_dim * final_layer["reduction_factor"])
            elif final_layer["type"] == "two_layers":
                self.final_layer = nn.SequentialCell(
                    nn.BatchNorm1d(self.out_dim), nn.ReLU(),
                    nn.Dense(self.out_dim, self.out_dim), nn.BatchNorm1d(self.out_dim), nn.ReLU(),
                    nn.Dense(self.out_dim, int(self.out_dim * final_layer["reduction_factor"]))
                )
                self.out_dim = int(self.out_dim * final_layer["reduction_factor"])

            else:
                raise ValueError("Unknown final layer type {}.".format(final_layer["type"]))

        else:
            self.final_layer = None

        """
        Here the normal method maybe something wrong 
        """

        for m in self.cells():
            if isinstance(m, nn.Conv2d):
                m.weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), m.weight.shape, ms.float32)
            elif isinstance(m, nn.Dense):
                m.weight = initializer(HeNormal(mode='fan_out', nonlinearity='relu'), m.weight.shape, ms.float32)

    def _make_layer(self, Block, planes, increase_dim=False, n=None):
        layers = []

        if increase_dim:
            layers.append(
                Block(
                    planes,
                    increase_dim=True,
                    last_relu=False,
                    downsampling=self._downsampling_type
                )
            )
            planes = 2 * planes

        for i in range(n):
            layers.append(Block(planes, last_relu=False, downsampling=self._downsampling_type))

        return Stage(layers, block_relu=self.last_relu)

    @property
    def last_conv(self):
        return self.stage_4.conv_b

    def construct(self, x):
        x = self.conv_1_3x3(x)  # x:(batch, ch, h, w)
        relu = ops.ReLU()
        x = relu(self.bn_1(x))

        feats_s1, x = self.stage_1(x)  # x:(batch, 16, h, w)  len of feats_s1: 5, shape: (batch, 16, 224, 224)
        feats_s2, x = self.stage_2(x)  # x:(batch, 32, h, w)  len of feats_s2: 5, shape: (batch, 32, 112, 112)
        feats_s3, x = self.stage_3(x)  # x:(batch, 64, h, w)  len of feats_s2: 4, shape: (batch, 64, 56, 56)
        x = self.stage_4(x)  # x:(batch, 64, h/4, w/4)

        raw_features = self.end_features(x)
        features = self.end_features(relu(x))

        # if self.all_attentions:
        #     attentions = [*feats_s1, *feats_s2, *feats_s3, x]
        # else:
        attentions = [feats_s1[-1], feats_s2[-1], feats_s3[-1], x]

        return [raw_features, features, attentions]
        # return {
        #     'raw_features': raw_features,
        #     'features': features,
        #     'attention': attentions
        # }

    def end_features(self, x):
        x = self.pool(x, (2, 3))
        x = x.view(x.shape[0], -1)

        if self.final_layer is not None:
            x = self.final_layer(x)

        return x


def resnet_rebuffi(n=5, **kwargs):
    return CifarResNet(n=n, **kwargs)


"""
Unit Test
    The result of x = resnet_rebuffi() + print(x) presents the same result as Pytorch, implies the network structure is right
    PYNATIVE_MODE is fine on Ascend;
    GRAPH_MODE is fine on CPU/Ascend 
    But is the momentum of BatchNorm is differently initialized
    dict return went wrong, we change it into list return
"""
# from mindspore import Tensor, context
# import numpy as np

#
# context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=1)
# context.set_context(mode=context.GRAPH_MODE, device_target='CPU') # CPU test
#
# net = resnet_rebuffi()
# y = Tensor(np.random.randn(128, 3, 256, 256).astype(np.float32))
# y = x(y)
# print(y)
# print(y[0].shape)  # (3, 64)
# print(y[1].shape)  # (3, 64)
# print(y[2][0].shape)  # (3, 16, 224, 224)
# print(y[2][1].shape)  # (3, 32, 112, 112)
# print(y[2][2].shape)  # (3, 64, 56, 56)

# conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
# no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
# group_params = [{'params': conv_params, 'grad_centralization': True},
#                 {'params': no_conv_params, 'lr': 0.01},
#                 {'order_params': net.trainable_params()}]
# # print(group_params)
# scheduler = nn.CosineDecayLR(min_lr=.0, max_lr=0.1, decay_steps=160)
# optimizer = nn.SGD(group_params, learning_rate=scheduler, weight_decay=0.01,
#                    momentum=0.9)
# print(optimizer)

#
# from mindspore.ops import stop_gradient
#
# logits = stop_gradient(net(y)[1])
# print(logits)
