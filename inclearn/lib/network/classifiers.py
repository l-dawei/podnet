import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.ops import stop_gradient
from mindspore.common.initializer import initializer, HeNormal, One
from mindspore import Tensor
from inclearn.lib import utils
from sklearn.cluster import KMeans
import logging
from mindspore.ops import stop_gradient

import numpy as np

from .postprocessors import FactorScalar
from inclearn.lib import distance as distance_lib

logger = logging.getLogger(__name__)

"""
    It seems that this class is to compute the similarity between tensors
    Unfinished, many TODO need to be fixed, and Unit Test is partly done
    
    5.31
    construct func, add_class, add_imprinted_class tested same value/shape on CPU as torch
    
    some torch.tensor.data is replaced with stop_gradient, but the paralist is not fixed
    
    some to.device func not fixed
    
    :return list[0](raw_features):  tensor (batch, 10) -> cifar    (not tested)
            list[1](features):      tensor (batch, 100) -> cifar
"""


class CosineClassifier(nn.Cell):
    classifier_type = "cosine"

    def __init__(
            self,
            features_dim,
            device,
            *,
            proxy_per_class=1,
            distance="cosine",
            merging="softmax",
            scaling=1,
            gamma=1.,
            use_bias=False,
            type=None,
            pre_fc=None,
            negative_weights_bias=None,
            train_negative_weights=False,
            eval_negative_weights=False
    ):
        super().__init__()

        self.n_classes = 0

        self.bias = None
        self.features_dim = features_dim
        self.proxy_per_class = proxy_per_class
        self.device = device
        self.distance = distance
        self.merging = merging
        self.gamma = gamma

        self.negative_weights_bias = negative_weights_bias
        self.train_negative_weights = train_negative_weights
        self.eval_negative_weights = eval_negative_weights

        self._negative_weights = None
        self.use_neg_weights = True
        # TODO: difference between pytorch's ParameterList and ms's ParameterTuple
        self._weights = ms.Parameter(
            initializer(HeNormal(nonlinearity='linear'), [self.proxy_per_class * self.n_classes, self.features_dim],
                        ms.float32))
        # self._weights = ms.ParameterTuple([])
        if isinstance(scaling, int) or isinstance(scaling, float):
            self.scaling = scaling
        else:
            logger.warning("Using inner learned scaling")
            self.scaling = FactorScalar(1.)

        if proxy_per_class > 1:
            logger.info("Using {} proxies per class.".format(proxy_per_class))

        if pre_fc is not None:
            self.pre_fc = nn.SequentialCell(
                nn.ReLU(),
                nn.BatchNorm1d(self.features_dim),
                nn.Dense(self.features_dim, pre_fc)
            )
            self.features_dim = pre_fc
        else:
            self.pre_fc = None

        self._task_idx = 0

        self._cmin = Tensor(0.0, ms.float32)
        self._cmax = Tensor(1000.0, ms.float32)

    def on_task_end(self):
        self._task_idx += 1
        if isinstance(self.scaling, nn.Cell):
            self.scaling.on_task_end()

    def on_epoch_end(self):
        if isinstance(self.scaling, nn.Cell):
            self.scaling.on_epoch_end()

    def construct(self, features):
        # if hasattr(self, "pre_fc") and self.pre_fc is not None:
        #     features = self.pre_fc(features)
        # features=Tensor(shape = (128, 64), dtype=ms.float32, init=One())
        logger.info('features shape {}'.format(features.shape))
        logger.info('features  {}'.format(type(features)))
        print('_negative_weights:', self._negative_weights)

        weights = self.weights()
        logger.info('weights length {}'.format(len(weights)))
        logger.info('weights type {}'.format(type(weights)))
        print('123456789')
        concat_op = ops.Concat()

        if self._negative_weights is not None and (
                self.training is True or self.eval_negative_weights
        ) and self.use_neg_weights:
            weights = concat_op(weights, self._negative_weights)

        # if self.distance == 'neg_stable_cosine_distance':
        l2_normalize = ops.L2Normalize(axis=-1)
        features = self.scaling * l2_normalize(features)
        weights = self.scaling * l2_normalize(weights)

        raw_similarities = -distance_lib.stable_cosine_distance(features, weights, self._cmin, self._cmax)

        # else:
        # print('Unknown distance function {}.')
        # raise NotImplementedError("Unknown distance function {}.".format(self.distance))

        if self.proxy_per_class > 1:
            similarities = self._reduce_proxies(raw_similarities)
        else:
            similarities = raw_similarities

        """
        It seems that in PODNet, the _negative_weights and negative_weights_bias is not used, so I didn't do this module
        """
        # if self._negative_weights is not None and self.negative_weights_bias is not None and self.training is True:
        #     qt = self._negative_weights.shape[0]

        # return a list instead of a dict
        logger.info('similarities {}.'.format(similarities))
        logger.info('raw_similarities {}.'.format(raw_similarities))
        return [similarities, raw_similarities]
        # return {"logits": similarities, "raw_logits": raw_similarities}

    def _reduce_proxies(self, similarities):
        # shape (batch_size, n_classes * proxy_per_class)
        n_classes = similarities.shape[1] / self.proxy_per_class

        # TODO: assert
        # assert n_classes.is_integer(), (similarities.shape[1], self.proxy_per_class)

        # n_classes = int(n_classes)
        bs = similarities.shape[0]

        # if self.merging == "softmax":
        #     simi_per_class = similarities.view(bs, n_classes, self.proxy_per_class)
        #     softmax = ops.Softmax(axis=-1)
        #     attentions = softmax(self.gamma * simi_per_class)
        #     return (simi_per_class * attentions).sum(-1)
        # else:
        #     raise ValueError("Unknown merging for multiple centers: {}.".format(self.merging))

        # bs=int(bs)
        # print('bs:',bs)
        n_classes = int(n_classes)
        print('n_classes:', n_classes)
        # print('self.proxy_per_class:',self.proxy_per_class)
        simi_per_class = similarities.view(bs, n_classes, self.proxy_per_class)
        softmax = ops.Softmax(axis=-1)
        attentions = softmax(self.gamma * simi_per_class)
        return (simi_per_class * attentions).sum(-1)

    # ------------------
    # Weights management
    # ------------------
    """
    It seems that in PODNet we didn't use any weight module, so I didn't implement some of them
    """

    # concat_op is fine with list operation
    # @property
    def weights(self):
        concat_op = ops.Concat()
        print('len(self._weights):', len(self._weights))
        return concat_op([clf for clf in self._weights])

    @property
    def new_weights(self):
        return self._weights[-1]

    @property
    def old_weights(self):
        if len(self._weights) > 1:
            return self._weights[:-1]
        return None

    def add_classes(self, n_classes):
        print('44444444444444444444444444444444')
        new_weights = ms.Parameter(
            initializer(HeNormal(nonlinearity='linear'), [self.proxy_per_class * n_classes, self.features_dim],
                        ms.float32))
        print('before:',self._weights)
        self._weights.append(new_weights)
        print('after:', self._weights)
        print(self._weights['shape'])

        self.n_classes += n_classes
        return self

    def add_imprinted_classes(
            self, class_indexes, inc_dataset, network, multi_class_diff="normal", type=None
    ):
        print('3333333333333333333333333333333333')
        if self.proxy_per_class > 1:
            logger.info("Multi class diff {}.".format(multi_class_diff))

        # TODO: weights_norm = self.weights.data.norm(dim=1, keepdim=True), use stop_gradient for detach?
        norm = nn.Norm(axis=1, keep_dims=True)
        print(type(norm(self.weights)))
        weights_norm = stop_gradient(norm(self.weights))
        avg_weights_norm = weights_norm.mean(axis=0)

        new_weights = []
        for class_index in class_indexes:
            _, loader = inc_dataset.get_custom_loader([class_index])
            features, _ = utils.extract_features(network, loader)

            l2_normalize1 = ops.L2Normalize(axis=1)
            features_normalized = l2_normalize1(ms.Tensor(features))

            mean_op = ops.ReduceMean(keep_dims=False)
            class_embeddings = mean_op(features_normalized, axis=0)

            l2_normalize2 = ops.L2Normalize(axis=0)
            class_embeddings = l2_normalize2(class_embeddings)

            if self.proxy_per_class == 1:
                new_weights.append(class_embeddings * avg_weights_norm)

            else:
                if multi_class_diff == "kmeans":
                    clusterizer = KMeans(n_clusters=self.proxy_per_class)
                    clusterizer.fit(features_normalized.asnumpy())

                    for center in clusterizer.cluster_centers_:
                        new_weights.append(ms.Tensor(center) * avg_weights_norm)
                else:
                    raise ValueError(
                        "Unknown multi class differentiation for imprinted weights: {}.".
                            format(multi_class_diff)
                    )
        stack = ops.Stack()
        new_weights = stack(new_weights)
        self._weights.append(ms.Parameter(new_weights))

        # self.to(self.device)
        self.n_classes += len(class_indexes)

        return self

    def set_negative_weights(self, negative_weights, ponderate=False):
        """Add weights that are used like the usual weights, but aren't actually
        parameters.

        :param negative_weights: Tensor of shape (n_classes * nb_proxy, features_dim)
        :param ponderate: Reponderate the negative weights by the existing weights norm, as done by
                          "Weights Imprinting".
        """
        logger.info("Add negative weights.")
        if isinstance(ponderate, str):
            if ponderate == "weights_imprinting":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                negative_weights = negative_weights * avg_weights_norm
            elif ponderate == "align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_weights_norm / avg_negative_weights_norm
                negative_weights = negative_weights * ratio
            elif ponderate == "inv_align_weights":
                avg_weights_norm = self.weights.data.norm(dim=1).mean()
                avg_negative_weights_norm = negative_weights.data.norm(dim=1).mean()

                ratio = avg_negative_weights_norm / avg_weights_norm
                negative_weights = negative_weights * ratio
            else:
                raise NotImplementedError(f"Unknown ponderation type {ponderate}.")

        if self.train_negative_weights:
            self._negative_weights = nn.Parameter(negative_weights)
        else:
            self._negative_weights = negative_weights