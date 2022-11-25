import logging
import os
import mindspore.dataset.vision.py_transforms as transforms
import multiprocessing
import mindspore as ms
import mindspore.dataset as ds
from mindspore.communication.management import init, get_rank, get_group_size
import numpy as np

logger = logging.getLogger(__name__)

"""
Some basic trans for ds are defined here
"""


class DataHandler:
    base_dataset = None
    train_transforms = []
    test_transforms = []
    common_transforms = []

    class_order = None
    open_image = False

    def set_custom_transforms(self, transforms):
        if transforms:
            raise NotImplementedError("Not implemented for modified transforms.")


class iCIFAR100(DataHandler):
    train_transforms = [
        ds.vision.c_transforms.RandomCrop((32, 32), (4, 4, 4, 4)),
        ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5),
        ds.vision.c_transforms.RandomColorAdjust(brightness=(63 / 255, 1)),  # FIXME: is this the same with Pytorch?
    ]

    common_transforms = [ds.vision.py_transforms.ToTensor(),
                         transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)), ]

    # Taken from original iCaRL implementation:
    class_order = [
        87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18,
        24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59,
        25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21,
        60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7,
        34, 55, 54, 26, 35, 39
    ]

    def set_custom_transforms(self, transforms):
        if not transforms.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)


class ImageNet1000(DataHandler):
    train_transforms = [
        ds.vision.c_transforms.RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333)),
        ds.vision.c_transforms.RandomHorizontalFlip(prob=0.5),
        ds.vision.c_transforms.RandomColorAdjust(brightness=(63 / 255, 1)),
    ]
    test_transforms = [
        ds.vision.c_transforms.Decode(),
        ds.vision.c_transforms.Resize(256),
        ds.vision.c_transforms.CenterCrop(224)
    ]

    mean = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    common_transforms = [ds.vision.c_transforms.Normalize(mean=mean, std=std), ds.vision.c_transforms.HWC2CHW()]
    imagenet_size = 1000

    open_image = True
    suffix = ""
    metadata_path = None

    def set_custom_transforms(self, transforms):
        if not transforms.get("color_jitter"):
            logger.info("Not using color jitter.")
            self.train_transforms.pop(-1)

    # def base_dataset(self, data_path, train=True):
    #     split = "train" if train else "val"
    #     print("Loading metadata of ImageNet_{} ({} split).".format(self.imagenet_size, split))
    #
    #     metadata_path = os.path.join(
    #         data_path if self.metadata_path is None else self.metadata_path,
    #         "{}_{}{}.txt".format(split, self.imagenet_size, self.suffix)
    #     )
    #
    #     self.data, self.targets = [], []
    #     with open(metadata_path) as f:
    #         for line in f:
    #             path, target = line.strip().split(" ")
    #
    #             self.data.append(os.path.join(data_path, path))
    #             self.targets.append(int(target))
    #
    #     self.data = np.array(self.data)
    #
    #     return self

# def _get_rank_info(distribute):
#     """
#     get rank size and rank id
#     """
#     if distribute:
#         init()
#         rank_id = get_rank()
#         device_num = get_group_size()
#     else:
#         rank_id = 0
#         device_num = 1
#     return device_num, rank_id
#
#
# def get_num_parallel_workers(num_parallel_workers):
#     """
#     Get num_parallel_workers used in dataset operations.
#     If num_parallel_workers > the real CPU cores number, set num_parallel_workers = the real CPU cores number.
#     """
#     cores = multiprocessing.cpu_count()
#     if isinstance(num_parallel_workers, int):
#         if cores < num_parallel_workers:
#             print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
#             num_parallel_workers = cores
#     else:
#         print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 8)))
#         num_parallel_workers = min(cores, 8)
#     return num_parallel_workers

# print(multiprocessing.cpu_count())  # eg: for HW is 72
# base_dataset = ds.Cifar100Dataset()
# print(base_dataset)
