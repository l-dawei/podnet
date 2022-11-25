from mindspore import Tensor
import mindspore.ops as ops
import mindspore.nn as nn


# TODO: Need Unit test
def pod(
        list_attentions_a,
        list_attentions_b,
        collapse_channels="spatial",
        normalize=True,
        memory_flags=None,
        only_old=False,
        **kwargs
):
    """
        Pooled Output Distillation.

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

    # TODO: loss = torch.tensor(0.).to(list_attentions_a[0].device), to
    loss = Tensor(0.)
    for i, (a, b) in enumerate(zip(list_attentions_a, list_attentions_b)):
        # shape of (b, n, w, h)
        assert a.shape == b.shape, (a.shape, b.shape)

        if only_old:
            a = a[memory_flags]
            b = b[memory_flags]
            if len(a) == 0:
                continue

        pow_op = ops.Pow()
        a = pow_op(a, 2)
        b = pow_op(b, 2)

        concat_op = ops.Concat(axis=-1)

        if collapse_channels == "spatial":
            a_h = a.sum(axis=3).view(a.shape[0], -1)
            b_h = b.sum(axis=3).view(b.shape[0], -1)
            a_w = a.sum(axis=2).view(a.shape[0], -1)
            b_w = b.sum(axis=2).view(b.shape[0], -1)
            a = concat_op([a_h, a_w])
            b = concat_op([b_h, b_w])

        else:
            raise ValueError("Unknown method to collapse: {}".format(collapse_channels))

        if normalize:
            l2_normalize = ops.L2Normalize(dim=1)
            a = l2_normalize(a)
            b = l2_normalize(b)

        mean = ops.ReduceMean(keep_dims=False)
        norm = nn.Norm(axis=-1)

        # torch.frobenius_norm output same with torch.norm
        layer_loss = mean(norm(a - b))
        loss += layer_loss

    return loss / len(list_attentions_a)
