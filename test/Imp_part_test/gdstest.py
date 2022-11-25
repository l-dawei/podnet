from PIL import Image
import mindspore.dataset as ds
import os
import numpy as np
from mindspore import Tensor


# import multiprocessing
#
# print(multiprocessing.cpu_count()) # 16 on laptop


class DummyDataset():
    def __init__(self, x, y, trsf):
        self.x = []
        for i in x:
            self.x.append(i)
        self.y = y
        self.trsf = trsf

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x, y = self.x[index], self.y[index]
        # x = Tensor(x)
        # self.trsf(x)
        return (x, y)
        # return {"inputs": x, "targets": y}


dataset_path = 'D:\\file\Dataset\\flower'
train_path = os.path.join(dataset_path, 'train')
val_path = os.path.join(dataset_path, 'val')
train_ds = ds.ImageFolderDataset(dataset_dir=train_path, num_parallel_workers=16)
val_ds = ds.ImageFolderDataset(dataset_dir=val_path, num_parallel_workers=16)
# train_ds.map(operations=ds.vision.c_transforms.Resize(224))
train_ds.batch(16)
print(train_ds.get_dataset_size())  # 3306
print(val_ds.get_dataset_size())  # 364
print(train_ds.get_batch_size())
x_train, y_train = [], []
for i, data in enumerate(train_ds.create_dict_iterator()):
    # print(data['image'].dtype)  # ms tensor UInt8
    # print(data['label'].dtype)  # ms tensor Int32
    # print(len(data['image']))
    # print(data['image'])
    x_train.append(data['image'].asnumpy())
    y_train.append(data['label'].asnumpy())

# Here will have some warning because we are create ndarray list of various each length
x_train = np.array(x_train, dtype=object)
y_train = np.array(y_train)
# if i >= 10:
#     break

img = Image.open('D:\\file\Dataset\\flower\\flower_photos\daisy\\5547758_eea9edfd54_n.jpg').convert("RGB")
# print(img)  # <PIL.Image.Image image mode=RGB size=320x232 at 0x1E897FEF208>

train_transforms = [
    ds.vision.c_transforms.RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
    ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5),
    ds.vision.c_transforms.RandomColorAdjust(brightness=(63 / 255, 1)),
]

mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
std = [0.229 * 255, 0.224 * 255, 0.225 * 255]
common_transforms = [ds.vision.c_transforms.Normalize(mean=mean, std=std), ds.vision.c_transforms.HWC2CHW()]
trsf = ds.transforms.c_transforms.Compose([*train_transforms, *common_transforms])

# print(x_train.shape)  # (3306,)
# print(y_train.shape)  # (3306,)
gds = ds.GeneratorDataset(source=DummyDataset(x_train, y_train, trsf), column_names=["image", "label"])
gds.batch(116, True)
# print(gds)
# x = gds.create_dict_iterator()
# print(x)
for i, data in enumerate(gds.create_dict_iterator()):  # i is from 0 to max_num
# print(i)
# print(data.keys())  # dict_keys(['image', 'label'])
    print(data['image'].shape)
# print(data['label'])
print(gds.get_dataset_size())
# print(gds.get_batch_size())
