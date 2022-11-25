import mindspore.ops as ops
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import numpy as np
from mindspore.ops import constexpr


# TODO: the output shape is fine(tensor(1)), but the precision is not fine!!!
# @constexpr
def nca(
        similarities,
        targets,
        class_weights=None,
        scale=1,
        margin=0.6,
):
    """
    Compute AMS cross-entropy loss.

    Reference:
        * Goldberger et al.
          Neighbourhood components analysis.
          NeuriPS 2005.
        * Feng Wang et al.
          Additive Margin Softmax for Face Verification.
          Signal Processing Letters 2018.

    :param similarities: Result of cosine similarities between weights and features.
    :param targets: Sparse targets.
    :param scale: Multiplicative factor, can be learned.
    :param margin: Margin applied on the "right" (numerator) similarities.
    :param memory_flags: Flags indicating memory samples, although it could indicate anything else.
    :return: A float scalar loss.

    eg: in cifar100's input, similarities is float32 tensor (128, 10), targets is int32 tensor (128, )

    """

    zeroslike = ops.ZerosLike()
    margins = zeroslike(similarities)

    # tmp = Tensor(np.arange(margins.shape[0]))

    margins[ms.numpy.arange(margins.shape[0]), targets.astype(ms.int32)] = margin
    # margins[tmp, targets.astype(ms.int32)] = margin

    similarities = scale * (similarities - margin)

    # print(similarities)

    # if exclude_pos_denominator:  # NCA-specific
    similarities = similarities - similarities.max(1)[0].view(-1, 1)  # Stability

    # print(similarities)

    disable_pos = zeroslike(similarities)

    # tmp = Tensor(np.arange(len(similarities)))

    disable_pos[ms.numpy.arange(len(similarities)), targets.astype(ms.int32)] = similarities[
        ms.numpy.arange(len(similarities)), targets.astype(ms.int32)]
    # disable_pos[tmp, targets.astype(ms.int32)] = similarities[
    #     tmp, targets.astype(ms.int32)]

    numerator = similarities[ms.numpy.arange(similarities.shape[0]), targets.astype(ms.int32)]
    # numerator = similarities[ms.numpy.arange(tmp), targets.astype(ms.int32)]

    denominator = similarities - disable_pos

    log = ops.Log()
    exp = ops.Exp()

    losses = numerator - log(exp(denominator).sum(-1))

    # print(losses)

    if class_weights is not None:
        losses = class_weights[targets] * losses

    losses = -losses
    # if hinge_proxynca:
    #     losses = ops.clip_by_value(losses, clip_value_min=Tensor(0., ms.float32),
    #                                clip_value_max=Tensor(1000., ms.float32))

    mean = ops.ReduceMean(keep_dims=False)
    loss = mean(losses)

    return loss


def embeddings_similarity(features_a, features_b):
    cosine_embedding_loss = nn.CosineEmbeddingLoss()
    ones = ops.Ones()
    return cosine_embedding_loss(
        features_a, features_b,
        ones(features_a.shape[0], ms.float32)
    )


"""
Unit Test
"""
# import numpy as np
#
# np.random.seed(1)
# x = np.random.randn(128, 10).astype(np.float32)
# y = np.random.randint(low=0, high=9, size=(128,), dtype=np.int32)
# x = Tensor(x)
# y = Tensor(y)
#
# # print(nca(x, y))  # tensor scalar 2.983711
# print(float(nca(x, y).asnumpy()))  # float 2.983711004257202
