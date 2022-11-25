import numpy as np
import mindspore.dataset as ds

"""
The needed cifar part of incdataset
"""
# dataset_path = "D://file//Dataset//cifar-100-binary"
# train_dataset = ds.Cifar100Dataset(dataset_path, usage='train')
# test_dataset = ds.Cifar100Dataset(dataset_path, usage='test')
# x_train, y_train = [], []
# for i, data in enumerate(train_dataset.create_dict_iterator()):
#     x_train.append(data['image'].asnumpy())  # asnumpy is because each one of data['image'] is a mindspore tensor
#     y_train.append(data['fine_label'])
#
# x_train = np.array(x_train)
# y_train = np.array(y_train)
# print(x_train.shape)
# print(y_train.shape)
"""
Test of part of ImageNet
"""
import os

#
# train = True
# split = "train" if train else "val"
# print("Loading metadata of ImageNet_{} ({} split).".format(1000, split))
# # Loading metadata of ImageNet_1000 (train split).
#
# metadata_path = None
# metadata_path = os.path.join(
#     'D:\\file\Dataset\\flower\\flower_photos' if metadata_path is None else metadata_path,
#     "{}_{}{}.txt".format(split, 1000, "")
# )
# print(metadata_path)  # D:\file\Dataset\flower\flower_photos\train_1000.txt
"""
image test
"""
# from PIL import Image
#
# op1 = ds.transforms.c_transforms.Compose([ds.vision.c_transforms.Resize(256)])
# x = x_train[1]
# # print(x.shape)
# img = Image.fromarray(x.astype("uint8"))
# img = op1(img)
# #
# # print(type(img))
# print(img.shape)
"""
The needed imagenet part of incdataset
"""
data_path = "D:\\file\Dataset\\flower"
train_dataset = ds.ImageFolderDataset(os.path.join(data_path, 'train'))
test_dataset = ds.ImageFolderDataset(os.path.join(data_path, 'val'))
#
trans = ds.transforms.c_transforms.Compose[ds.vision.c_transforms.Decode(),
                                           ds.vision.c_transforms.Resize(224),]
#
# trans1 = [ds.vision.c_transforms.RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
#           ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5)]
#
# train_dataset.map(operations=trans,
#                   input_columns='image')
# # train_dataset.map(operations=ds.vision.c_transforms.Resize(224), )
x_train, y_train = [], []
#
for i, data in enumerate(train_dataset.create_dict_iterator()):
    # x_train.append(data['image'].asnumpy())
    # y_train.append(data['label'].asnumpy())
    # print(data['image'].shape)
    x_train.append(data['image'])

# x_train = np.array(x_train)
# y_train = np.array(y_train)

# print(len(x_train))  # 3306
# print(len(y_train))  # 3306
# print(x_train[0].shape)  # (72725,)
# print(x_train[0].dtype)  # uint8
# print(type(y_train[0]))  # <class 'numpy.ndarray'>  np整形
# print(y_train[0].dtype)  # int32

# print(x_train)
# print(x_train.shape)  # (3306,)
# print(y_train.shape)  # (3306,)
xxx = trans(x_train[0])
print(xxx)
