"""
This is for testing the ImageNet dataset on Ascend Server
"""
import mindspore.dataset as ds
import multiprocessing
import os


def get_num_parallel_workers(num_parallel_workers):
    """
    Get num_parallel_workers used in dataset operations.
    If num_parallel_workers > the real CPU cores number, set num_parallel_workers = the real CPU cores number.
    """
    cores = multiprocessing.cpu_count()
    if isinstance(num_parallel_workers, int):
        if cores < num_parallel_workers:
            print("The num_parallel_workers {} is set too large, now set it {}".format(num_parallel_workers, cores))
            num_parallel_workers = cores
    else:
        print("The num_parallel_workers {} is invalid, now set it {}".format(num_parallel_workers, min(cores, 8)))
        num_parallel_workers = min(cores, 8)
    return num_parallel_workers


# print(get_num_parallel_workers(12))
# print(multiprocessing.cpu_count())  eg: test is 72

dataset_path = '/disk0/dataset/imagenet/'
train_path = os.path.join(dataset_path, 'train')  # /disk0/dataset/imagenet/train
val_path = os.path.join(dataset_path, 'val')  # /disk0/dataset/imagenet/train
train_ds = ds.ImageFolderDataset(dataset_dir=train_path, num_parallel_workers=72)
val_ds = ds.ImageFolderDataset(dataset_dir=val_path, num_parallel_workers=72)
print(train_ds.get_dataset_size())  # 1281167
print(val_ds.get_dataset_size())  # 50000

for i, data in enumerate(train_ds.create_dict_iterator()):
    print(data['label'])

    if i >= 10:
        break
