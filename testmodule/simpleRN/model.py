import mindspore.nn as nn
from mindspore.common.tensor import Tensor

class ResidualBlockBase(nn.Cell):
    """
    ResNet V1 residual block definition.

    Args:
        in_channel (int): Input channel.
        out_channel (int): Output channel.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        res_base (bool): Enable parameter setting of resnet18. Default: True.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ResidualBlockBase(3, 256, stride=2)
    """

    def __init__(self,
                 in_channel,
                 out_channel,
                 stride=1,
                 res_base=True):
        super(ResidualBlockBase, self).__init__()
        self.res_base = res_base
