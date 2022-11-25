import mindspore as ms
import mindspore.nn as nn

"""
Weldon Pooling is a critical module developped in the frame of a conference paper published 
at CVPR 2016 "WELDON: Weakly Supervised Learning of Deep Convolutional Neural Networks".
https://github.com/Cadene/weldon.torch

This file is desperated because we didn't use this
"""


class WeldonPool2d(nn.Cell):
    def __init__(self, kmax=1, kmin=None, **kwargs):
        super(WeldonPool2d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax

        print("Using Weldon Pooling with kmax={}, kmin={}.".format(self.kmax, self.kmin))
        self._pool_func = self._define_function()

    def construct(self, input):
        return self._pool_func(input)

    """
    Here we need to make a new pooling function based on the ms's module. so this is uncompleted
    """

    def _define_function(self):
        class WeldonPool2dFunction():
            @staticmethod
            def get_number_of_instances(k, n):
                if k <= 0:
                    return 0
                elif k < 1:
                    return round(k * n)
                elif k > n:
                    return int(n)
                else:
                    return int(k)

            @staticmethod
            def construct(ctx, input):
                # get batch information
                batch_size = input.size(0)
                num_channels = input.size(1)
                h = input.size(2)
                w = input.size(3)

                # get number of regions
                n = h * w

                # get the number of max and min instances
                kmax = WeldonPool2dFunction.get_number_of_instances(self.kmax, n)
                kmin = WeldonPool2dFunction.get_number_of_instances(self.kmin, n)

                # sort scores (no out for ms's sort, i don't know if this may cause anything unexpected)
                # sorted, indices = input.new(), input.new().long()
                # ms.ops.sort(input.view(batch_size, num_channels, n), axis=2, descending=True)

                # compute scores for max instances
                # indices_max = indices.narrow(2, 0, kmax)
                output = sorted.narrow(2, 0, kmax).sum(2).div_(kmax)

                if kmin > 0:
                    # compute scores for min instances
                    # indices_min = indices.narrow(2, n-kmin, kmin)
                    output.add_(sorted.narrow(2, n - kmin, kmin).sum(2).div_(kmin)).div_(2)

                # return output with right size
                return output.view(batch_size, num_channels)

    def __repr__(self):
        return self.__class__.__name__ + ' (kmax=' + str(self.kmax
                                                         ) + ', kmin=' + str(self.kmin) + ')'
