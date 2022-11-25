import mindspore.dataset as ds
import mindspore as ms
import numpy as np
import os

dataset_path1 = 'D://file//Dataset//flower'
dataset_path2 = "D://file//Dataset//cifar-100-binary"

"""
ImageNet part(use laptop for test, so flower ds instead)

ImageFolderDataset has two colomn: image and label

"""
train_ds = ds.ImageFolderDataset(dataset_dir=os.path.join(dataset_path1, 'train'))
val_ds = ds.ImageFolderDataset(dataset_dir=os.path.join(dataset_path1, 'val'))
# print(train_ds.get_dataset_size())  # 3306
# print(val_ds.get_dataset_size())  # 364

mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
train_trans1 = [ds.vision.c_transforms.RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
                ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5),
                ds.vision.c_transforms.RandomColorAdjust(brightness=(63 / 255, 1)),
                ds.vision.c_transforms.Normalize(mean=mean, std=std),
                ds.vision.c_transforms.HWC2CHW()]
test_trans1 = [
    ds.vision.c_transforms.Decode(),
    ds.vision.c_transforms.Resize(256),
    ds.vision.c_transforms.CenterCrop(224),
    ds.vision.c_transforms.Normalize(mean=mean, std=std),
    ds.vision.c_transforms.HWC2CHW()]

train_ds = train_ds.map(operations=train_trans1, input_columns="image")
train_ds = train_ds.batch(128)
# print(train_ds.get_batch_size())
# data is a dict of keys {'image', 'label'}
for i, data in enumerate(train_ds.create_dict_iterator()):
    if i >= 10:
        break
    print(data['image'].shape)
#     """
#     Why the shape still so?
#     without any trans   after train_trans
#     (36056,)            (34945,)
#     (21710,)            (30747,)
#     (63778,)            (33071,)
#     (36398,)            (60867,)
#     (21667,)            (186026,)
#     (112559,)           (34293,)
#     (28600,)            (22795,)
#     (42464,)            (62858,)
#     (111574,)           (17454,)
#     (19641,)            (92862,)
#     """

"""
Cifar part
"""
# train_dataset = ds.Cifar100Dataset(dataset_path2, usage='train')
# test_dataset = ds.Cifar100Dataset(dataset_path2, usage='test')
# print(train_dataset.get_dataset_size())  # 50000
# print(test_dataset.get_dataset_size())  # 10000

train_trans2 = [ds.vision.c_transforms.RandomCrop((32, 32), (4, 4, 4, 4)),
                ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5),
                ds.vision.c_transforms.RandomColorAdjust(brightness=(63 / 255, 1)),
                ds.vision.c_transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
                ds.vision.c_transforms.HWC2CHW()]
test_trans2 = []
# for i, data in enumerate(train_dataset.create_dict_iterator()):
#     if i >= 10:
#         break
#     print(data['image'].shape)  # (32, 32, 3)

# train_dataset.batch(20)
# print(train_dataset.get_batch_size())  # I set it 20, but why output 1 ???
"""
Customize ds part
"""


class DummyDataset():
    def __init__(self, x, y, memory_flags):
        self.y = y
        self.memory_flags = memory_flags
        if x.dtype == object:
            self.x = []
            for i in x:
                self.x.append(i)
            assert len(x) == y.shape[0] == memory_flags.shape[0]
        else:
            self.x = x
            assert x.shape[0] == y.shape[0] == memory_flags.shape[0]

    def __len__(self):
        if isinstance(self.x, list):
            return len(self.x)
        else:
            return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        memory_flag = self.memory_flags[idx]

        return x, y, memory_flag


# x_train1, y_train1 = [], []
# for i, data in enumerate(train_ds.create_dict_iterator()):
#     x_train1.append(data['image'].asnumpy())
#     y_train1.append(data['label'].asnumpy())
#
# x_train1 = np.array(x_train1)
# y_train1 = np.array(y_train1)
#
# x_train2, y_train2 = [], []
# for i, data in enumerate(train_dataset.create_dict_iterator()):
#     x_train2.append(data['image'].asnumpy())
#     y_train2.append(data['fine_label'].asnumpy())
#
# x_train2 = np.array(x_train2)
# y_train2 = np.array(y_train2)


# gds = ds.GeneratorDataset(
#     source=DummyDataset(x_train1, y_train1, np.zeros((x_train1.shape[0],))),
#     column_names=["image", "label", "memory_flag"],
# )
# gds = gds.map(operations=train_trans1, input_columns="image",
#               )
# gds = gds.map(operations=ds.transforms.c_transforms.TypeCast(ms.float32), input_columns="image",
#               )
# gds = gds.map(operations=ds.transforms.c_transforms.TypeCast(ms.int32), input_columns="label",
#               )
# gds = gds.map(operations=ds.transforms.c_transforms.TypeCast(ms.int32), input_columns="memory_flag",
#               )
# gds = gds.batch(128)
#
# for i, data in enumerate(gds.create_dict_iterator()):
#     if i >= 10:
#         break
# print(type(data["image"]))  # all ms tensor Float32
# print(data["image"].shape)  # (128, 3, 224, 224) vs (128, 3, 32, 32)
# print(type(data["label"]))  # all ms tensor Int32
# print(data["label"].shape)  # (128,)
# print(data["memory_flag"].shape) # (128,)

def _map_new_class_index(y, order):
    """Transforms targets for new class order."""
    return np.array(list(map(lambda x: order.index(x), y)))


order = [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11,
         4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82,
         53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2,
         95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]

# x = _map_new_class_index(y_train1, order)
# print(x)

