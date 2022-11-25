import numpy as np
import mindspore as ms
from mindspore import Tensor
import mindspore.nn as nn
import mindspore.ops as ops
from mindspore.common.initializer import initializer, HeNormal

"""
Test on main part of Cosine classifier, this is responding to newcode's test8 
"""
np.random.seed(1)
weights = Tensor(np.random.randn(48, 64), ms.float32)
features = Tensor(np.random.randn(32, 64), ms.float32)
#
# l2_normalize = ops.L2Normalize(axis=-1)
# features = 1 * l2_normalize(features)
# weights = 1 * l2_normalize(weights)
# # print(features.shape)  # (32, 64)
# print(features)
# # print(weights.shape)  # (32, 64)
# print(weights)
#
# raw_similarities = Tensor(np.random.randn(32, 32), ms.float32)
proxy_per_class = 8
# n_classes = raw_similarities.shape[1] / proxy_per_class  # tensor scaler
# n_classes = int(n_classes)
# # print(n_classes)  # int, 32/8
#
# simi_per_class = raw_similarities.view(32, n_classes, proxy_per_class)
# softmax = ops.Softmax(axis=-1)
# attentions = softmax(1. * simi_per_class)
# print(attentions.shape)  # 32, 4, 8
#
# re0 = (simi_per_class * attentions).sum(-1)
#
# print(re0.shape)
# print(type(re0))
"""
Test of add_classes

"""
# new_weights = ms.Parameter(
#     initializer(HeNormal(nonlinearity='linear'), [32, 128],
#                 ms.float32))
#
# print(new_weights)
"""
Test of add_imprinted_classes
"""
from sklearn.cluster import KMeans

norm = nn.Norm(axis=1, keep_dims=True)
weights_norm = norm(weights)
avg_weights_norm = weights_norm.mean(axis=0)
# print(weights_norm.shape)  # ms tensor (48, 1)
# print(type(avg_weights_norm))
# print(avg_weights_norm.shape)  # ms tensor (1, )
# print(avg_weights_norm)

l2_normalize1 = ops.L2Normalize(axis=1)
features_normalized = l2_normalize1(features)

mean_op = ops.ReduceMean(keep_dims=False)
class_embeddings = mean_op(features_normalized, axis=0)

l2_normalize2 = ops.L2Normalize(axis=0)
class_embeddings = l2_normalize2(class_embeddings)

# print(class_embeddings.shape)  # (64,)
# print(class_embeddings)

# tmp = class_embeddings * avg_weights_norm
# print(tmp.shape)  # 64

new_weights = []

clusterizer = KMeans(n_clusters=proxy_per_class)
clusterizer.fit(features_normalized.asnumpy())

# proxy_per_class times
for center in clusterizer.cluster_centers_:
    # tmp = ms.Tensor(center) * avg_weights_norm
    # print(tmp.shape)  # (64,)
    new_weights.append(class_embeddings * avg_weights_norm)

stack = ops.Stack()
new_weights = stack(new_weights)
print(new_weights.shape)  # (8, 64)
