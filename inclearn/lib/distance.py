import mindspore
import mindspore.ops as ops
from mindspore import Tensor

"""
5.26
Test succeed on both cpu and Ascend with Graphmode, same value and shape with Pytorch 
:return: ms.tensor
"""


def stable_cosine_distance(a, b, _cmim, _cmax):
    """Computes the pairwise distance matrix with numerical stability."""
    concat_op = ops.Concat()
    mat = concat_op((a, b))

    add_op = ops.Add()
    pow_op = ops.Pow()
    sum_op = ops.ReduceSum(keep_dims=True)
    expand_op0 = ops.BroadcastTo((mat.shape[0], -1))
    expand_op1 = ops.BroadcastTo((mat.shape[0], -1))

    mm_op = ops.MatMul()
    # tm1 = expand_op(sum_op(pow_op(mat, 2), axis=1))
    # tmp2 = expand_op(sum_op(pow_op(mat.T, 2), axis=0))
    ep1 = expand_op0(sum_op(pow_op(mat, 2), 1))
    sum_op = ops.ReduceSum(keep_dims=True)
    ep2 = expand_op1(sum_op(pow_op(mat.T, 2), 0))
    pairwise_distances_squared = add_op(ep1, ep2) - 2 * mm_op(mat, mat.T)

    # _clip_value_min = Tensor(0.0, mindspore.float32)
    # _clip_value_max = Tensor(1000.0, mindspore.float32)
    # Deal with numerical inaccuracies. Set small negatives to zero.
    # TODO: However, we don't know the max of ms's ops.clip_by_value, while in Pytorch just min is okay, so just set max to 1k, but I don't know if this is okay
    pairwise_distances_squared = ops.clip_by_value(pairwise_distances_squared,
                                                   clip_value_min=_cmim,
                                                   clip_value_max=_cmax)

    # Get the mask where the zero distances are at.
    le_op = ops.LessEqual()
    error_mask = le_op(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    # if squared:
    pairwise_distances = pairwise_distances_squared
    # else:
    #     sqrt_op = ops.Sqrt()
    #     pairwise_distances = sqrt_op(pairwise_distances_squared + error_mask.astype(mindspore.float32) * 1e-6)

    # Undo conditionally adding 1e-16.
    mul_op = ops.Mul()
    pairwise_distances = mul_op(pairwise_distances, (error_mask == False).astype(mindspore.float32))

    # Explicitly set diagonals to zero.
    eye_op = ops.Eye()
    mask_offdiagonals = 1 - eye_op(pairwise_distances.shape[0], pairwise_distances.shape[1], mindspore.float32)
    pairwise_distances = mul_op(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]


"""
Unit Test
"""
# import numpy as np
#
# from mindspore import context
#
# context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=1)
# context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

# np.random.seed(5)
#
# x = np.random.randn(128, 64)
# y = np.random.randn(228, 64)
# x = Tensor(np.random.randn(128, 64), mindspore.float32)
# y = Tensor(np.random.randn(228, 64), mindspore.float32)
# print(x, y)
# a = stable_cosine_distance(x, y)
# print(a.shape)
