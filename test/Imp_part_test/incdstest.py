# from inclearn.lib.data.datasets import iCIFAR100
import mindspore.dataset as ds
import numpy as np

"""
First part test
"""


def _split_per_class(x, y, validation_split=0.):
    """Splits train data for a subset of validation data.

    Split is done so that each class has a much data.
    """
    shuffled_indexes = np.random.permutation(x.shape[0])  # a shuffled index ndarray of size 50000 for cifar
    x = x[shuffled_indexes]  # only change the order of classes
    y = y[shuffled_indexes]  # only change the order of classes

    x_val, y_val = [], []
    x_train, y_train = [], []

    for class_id in np.unique(y):
        class_indexes = np.where(y == class_id)[0]
        nb_val_elts = int(class_indexes.shape[0] * validation_split)

        val_indexes = class_indexes[:nb_val_elts]
        train_indexes = class_indexes[nb_val_elts:]

        x_val.append(x[val_indexes])
        y_val.append(y[val_indexes])
        x_train.append(x[train_indexes])
        y_train.append(y[train_indexes])

    x_train, y_train = np.concatenate(x_train), np.concatenate(y_train)
    x_val, y_val = np.concatenate(x_val), np.concatenate(y_val)
    return x_val, y_val, x_train, y_train


data_path = "D://file//Dataset//cifar-100-binary"
train_dataset = ds.Cifar100Dataset(data_path, usage='train')
test_dataset = ds.Cifar100Dataset(data_path, usage='test')

x_train, y_train = [], []

for i, data in enumerate(train_dataset.create_dict_iterator()):
    x_train.append(data['image'].asnumpy())
    y_train.append(data['fine_label'].asnumpy())

x_train = np.array(x_train)
y_train = np.array(y_train)

x_val, y_val, x_train, y_train = _split_per_class(
    x_train, y_train, .0
)

# x_test, y_test = [], []
#
# for i, data in enumerate(test_dataset.create_dict_iterator()):
#     x_test.append(data['image'].asnumpy())
#     y_test.append(data['fine_label'])
#
# x_test = np.array(x_train)
# y_test = np.array(y_train)

order = list(range(len(np.unique(y_train))))
order1 = [[87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11,
           4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82,
           53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85,
           2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39],
          [58, 30, 93, 69, 21, 77, 3, 78, 12, 71, 65, 40, 16, 49, 89, 46, 24, 66, 19, 41, 5, 29, 15, 73, 11, 70, 90, 63,
           67, 25, 59, 72, 80, 94, 54, 33, 18, 96, 2, 10, 43, 9, 57, 81, 76, 50, 32, 6, 37, 7, 68, 91, 88, 95, 85, 4,
           60, 36, 22, 27, 39, 42, 34, 51, 55, 28, 53, 48, 38, 17, 83, 86, 56, 35, 45, 79, 99, 84, 97, 82, 98, 26, 47,
           44, 62, 13, 31, 0, 75, 14, 52, 74, 8, 20, 1, 92, 87, 23, 64, 61],
          [71, 54, 45, 32, 4, 8, 48, 66, 1, 91, 28, 82, 29, 22, 80, 27, 86, 23, 37, 47, 55, 9, 14, 68, 25, 96, 36, 90,
           58, 21, 57, 81, 12, 26, 16, 89, 79, 49, 31, 38, 46, 20, 92, 88, 40, 39, 98, 94, 19, 95, 72, 24, 64, 18, 60,
           50, 63, 61, 83, 76, 69, 35, 0, 52, 7, 65, 42, 73, 74, 30, 41, 3, 6, 53, 13, 56, 70, 77, 34, 97, 75, 2, 17,
           93, 33, 84, 99, 51, 62, 87, 5, 15, 10, 78, 67, 44, 59, 85, 43, 11]]

# 100 of each
print(len(order1[0]))
print(len(order1[1]))
print(len(order1[2]))

# print(x_val.shape)
# print(x_val)
# print(y_val.shape)
# print(y_val)
# print(x_train.shape) #(50000, 32, 32, 3)
# print(y_train.shape) #(50000,)
# print(order)

"""
Small test
"""
# x = np.random.permutation(50000)
# # print(x.shape)
# y = np.random.randn(50000, 32, 32, 3)
# y = y[x]
# print(y.shape)
# y = np.array([1, 2, 3, 4, 5])
# x = np.array([1, 0, 3, 2, 4])
# print(y[x])
