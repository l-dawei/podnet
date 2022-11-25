import functools
import torch
import torch.nn.functional as F
import mindspore as ms
from mindspore import Tensor
from mindspore.common.initializer import One
import mindspore.ops as ops


def f_norm2(x):
    res = ms.Tensor(shape=(x.shape[0],),dtype=ms.float32,init=One())
    print(res)
    for i in range(x.shape[0]):
        temp=ops.pow(ops.sum(x[i]*x[i]),0.5)
        print(temp)
        res[i]=temp
        # print(res[i])
    # print(res[0])
    # print(res)
    return res

def pod(
    list_attentions_a,
    list_attentions_b,
    collapse_channels="spatial",
    normalize=True,
    memory_flags=None,
    only_old=False,
    **kwargs
):
    """Pooled Output Distillation.

    Reference:
        * Douillard et al.
          Small Task Incremental Learning.
          arXiv 2020.

    :param list_attentions_a: A list of attention maps, each of shape (b, n, w, h).
    :param list_attentions_b: A list of attention maps, each of shape (b, n, w, h).
    :param collapse_channels: How to pool the channels.
    :param memory_flags: Integer flags denoting exemplars.
    :param only_old: Only apply loss to exemplars.
    :return: A float scalar loss.
    """
    assert len(list_attentions_a) == len(list_attentions_b)

    # loss = ms.Tensor(0.).to(list_attentions_a[0].device)
    loss =ms.Tensor(0.)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        a = ops.pow(a, 2)
        b = ops.pow(b, 2)

        if collapse_channels == "channels":
            a = a.sum(axis=1).view(a.shape[0], -1)  # shape of (b, w * h)
            b = b.sum(axis=1).view(b.shape[0], -1)
        elif collapse_channels == "width":
            a = a.sum(axis=2).view(a.shape[0], -1)  # shape of (b, c * h)
            b = b.sum(axis=2).view(b.shape[0], -1)
        elif collapse_channels == "height":
            a = a.sum(axis=3).view(a.shape[0], -1)  # shape of (b, c * w)
            b = b.sum(axis=3).view(b.shape[0], -1)
        elif collapse_channels == "gap":
            a = ops.AdaptiveAvgPool2D(a, (1, 1))[..., 0, 0]
            b = ops.AdaptiveAvgPool2D(b, (1, 1))[..., 0, 0]
        elif collapse_channels == "spatial":
            a_h = a.sum(axis=3).view(a.shape[0], -1)
            b_h = b.sum(axis=3).view(b.shape[0], -1)
            a_w = a.sum(axis=2).view(a.shape[0], -1)
            b_w = b.sum(axis=2).view(b.shape[0], -1)
            concat_op = ops.Concat(axis=-1)
            cast_op = ops.Cast()
            a = concat_op((a_h, a_w))
            b = concat_op((b_h, b_w))
        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            l2norm = ms.ops.L2Normalize(axis=1, epsilon=1e-12)
            a = l2norm(a)
            b = l2norm(b)
        print('a:',a)
        print('b:',b)
        op=ops.ReduceMean(keep_dims=True)
        c=a-b
        layer_loss = op(f_norm2(c),-1)
        loss += layer_loss

    return loss / len(list_attentions_a)


def perceptual_features_reconstruction(list_attentions_a, list_attentions_b, factor=1.):
    loss = 0.

    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        bs, c, w, h = a.shape

        # a of shape (b, c, w, h) to (b, c * w * h)
        a = a.view(bs, -1)
        b = b.view(bs, -1)

        a = ms.ops.L2Normalize(a, p=2, dim=-1)
        b = ms.ops.L2Normalize(b, p=2, dim=-1)

        layer_loss = (F.pairwise_distance(a, b, p=2)**2) / (c * w * h)
        loss += ms.ops.ReduceMean(layer_loss)

    return factor * (loss / len(list_attentions_a))


def perceptual_style_reconstruction(list_attentions_a, list_attentions_b, factor=1.):
    loss = 0.

    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        bs, c, w, h = a.shape

        a = a.view(bs, c, w * h)
        b = b.view(bs, c, w * h)

        gram_a = ms.ops.BatchMatMul(a, a.transpose(2, 1)) / (c * w * h)
        gram_b = ms.ops.BatchMatMul(b, b.transpose(2, 1)) / (c * w * h)

        layer_loss = torch.frobenius_norm(gram_a - gram_b, dim=(1, 2))**2

        # layer_loss = f_norm2(gram_a - gram_b, dim=(1, 2))**2
        loss += layer_loss.mean()

    return factor * (loss / len(list_attentions_a))


def gradcam_distillation(gradients_a, gradients_b, activations_a, activations_b, factor=1):
    """Distillation loss between gradcam-generated attentions of two models.

    References:
        * Dhar et al.
          Learning without Memorizing
          CVPR 2019

    :param base_logits: [description]
    :param list_attentions_a: [description]
    :param list_attentions_b: [description]
    :param factor: [description], defaults to 1
    :return: [description]
    """
    attentions_a = _compute_gradcam_attention(gradients_a, activations_a)
    attentions_b = _compute_gradcam_attention(gradients_b, activations_b)

    assert len(attentions_a.shape) == len(attentions_b.shape) == 4
    assert attentions_a.shape == attentions_b.shape

    batch_size = attentions_a.shape[0]

    flat_attention_a = ms.ops.L2Normalize(attentions_a.view(batch_size, -1), p=2, dim=-1)
    flat_attention_b = ms.ops.L2Normalize(attentions_b.view(batch_size, -1), p=2, dim=-1)

    distances = ms.ops.Abs(flat_attention_a - flat_attention_b).sum(-1)

    return factor * ms.ops.ReduceMean(distances)


def _compute_gradcam_attention(gradients, activations):
    alpha = ms.ops.AdaptiveAvgPool2D(gradients, (1, 1))
    return ms.ops.ReLU(alpha * activations)


@functools.lru_cache(maxsize=1, typed=False)
def _get_mmd_factor(sigmas, device):
    sigmas = ms.Tensor(sigmas)[:, None, None].to(device).float()
    sigmas = -1 / (2 * sigmas)
    return sigmas