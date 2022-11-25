import logging
import random
from inclearn.lib.data.datasets import iCIFAR100, ImageNet1000
import numpy as np
import mindspore.dataset as ds
import multiprocessing
import os
import mindspore as ms
from mindspore.communication import get_rank, get_group_size

logger = logging.getLogger(__name__)


class IncrementalDataset:
    """
    Incremental generator of datasets.

    :param dataset_name: Among a list of available dataset, that can easily be defined (see at file's end).
    :param random_order: Shuffle the class ordering, else use a cherry-picked ordering.
    :param shuffle: Shuffle batch order between epochs.
    :param workers: Number of workers loading the data.
    :param batch_size: The batch size.
    :param seed: Seed to force determinist class ordering.
    :param increment: Number of class to add at each task.
    :param validation_split: Percent of training data to allocate for validation.
    :param onehot: Returns targets encoded as onehot vectors instead of scalars.
                   Memory is expected to be already given in an onehot format.
    :param initial_increment: Initial increment may be defined if you want to train on more classes than usual
                    for the first task, like UCIR does.
    """

    def __init__(
            self,
            dataset_name,
            random_order=False,
            shuffle=True,
            workers=16,
            batch_size=128,
            seed=1,
            increment=10,
            validation_split=0.,
            onehot=False,
            initial_increment=None,
            sampler=None,
            sampler_config=None,
            data_path="data",
            class_order=None,
            dataset_transforms=None,
            all_test_classes=False,
            metadata_path=None
    ):
        datasets = _get_datasets(dataset_name)  # eg: <class 'datasets.iCIFAR100'>
        if metadata_path:
            print("Adding metadata path {}".format(metadata_path))
            datasets[0].metadata_path = metadata_path

        self._workers = workers
        self._setup_data(
            datasets,
            random_order=random_order,
            class_order=class_order,
            seed=seed,
            increment=increment,
            validation_split=validation_split,
            initial_increment=initial_increment,
            data_path=data_path
        )

        dataset = datasets[0]()
        # dataset.set_custom_transforms(dataset_transforms)
        self.train_transforms = dataset.train_transforms  # FIXME handle multiple datasets
        self.test_transforms = dataset.test_transforms
        self.common_transforms = dataset.common_transforms

        self.open_image = datasets[0].open_image

        self._current_task = 0

        self._seed = seed
        self._batch_size = batch_size

        self._shuffle = shuffle
        self._onehot = onehot
        self._sampler = sampler
        self._sampler_config = sampler_config
        self._all_test_classes = all_test_classes

    # for cifar is 51, for imagenet is 501
    @property
    def n_tasks(self):
        return len(self.increments)

    # total class num, for cifar is 100, for imagenet is 1k
    @property
    def n_classes(self):
        return sum(self.increments)

    def new_task(self, memory=None, memory_val=None):
        if self._current_task >= len(self.increments):
            raise Exception("No more tasks.")

        min_class = sum(self.increments[:self._current_task])
        max_class = sum(self.increments[:self._current_task + 1])

        x_train, y_train = self._select(
            self.data_train, self.targets_train, low_range=min_class, high_range=max_class
        )
        nb_new_classes = len(np.unique(y_train))
        x_val, y_val = self._select(
            self.data_val, self.targets_val, low_range=min_class, high_range=max_class
        )
        if self._all_test_classes is True:
            logger.info("Testing on all classes!")
            x_test, y_test = self._select(
                self.data_test, self.targets_test, high_range=sum(self.increments)
            )
        elif self._all_test_classes is not None or self._all_test_classes is not False:
            max_class = sum(self.increments[:self._current_task + 1 + self._all_test_classes])
            logger.info(
                f"Testing on {self._all_test_classes} unseen tasks (max class = {max_class})."
            )
            x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)
        else:
            x_test, y_test = self._select(self.data_test, self.targets_test, high_range=max_class)

        if self._onehot:
            def to_onehot(x):
                n = np.max(x) + 1
                return np.eye(n)[x]

            y_train = to_onehot(y_train)

        if memory is not None:
            logger.info("Set memory of size: {}.".format(memory[0].shape[0]))
            x_train, y_train, train_memory_flags = self._add_memory(x_train, y_train, *memory)
        else:
            train_memory_flags = np.zeros((x_train.shape[0],))
        if memory_val is not None:
            logger.info("Set validation memory of size: {}.".format(memory_val[0].shape[0]))
            x_val, y_val, val_memory_flags = self._add_memory(x_val, y_val, *memory_val)
        else:
            val_memory_flags = np.zeros((x_val.shape[0],))

        train_loader = self._get_loader(x_train, y_train, train_memory_flags, mode="train")
        val_loader = self._get_loader(x_val, y_val, val_memory_flags,
                                      mode="train") if len(x_val) > 0 else None
        test_loader = self._get_loader(x_test, y_test, np.zeros((x_test.shape[0],)), mode="test")

        task_info = {
            "min_class": min_class,
            "max_class": max_class,
            "total_n_classes": sum(self.increments),
            "increment": nb_new_classes,  # self.increments[self._current_task],
            "task": self._current_task,
            "max_task": len(self.increments),
            "n_train_data": x_train.shape[0],
            "n_test_data": x_test.shape[0]
        }

        self._current_task += 1

        return task_info, train_loader, val_loader, test_loader

    def _add_memory(self, x, y, data_memory, targets_memory):
        if self._onehot:  # Need to add dummy zeros to match the number of targets:
            targets_memory = np.concatenate(
                (
                    targets_memory,
                    np.zeros((targets_memory.shape[0], self.increments[self._current_task]))
                ),
                axis=1
            )

        memory_flags = np.concatenate((np.zeros((x.shape[0],)), np.ones((data_memory.shape[0],))))

        x = np.concatenate((x, data_memory))
        y = np.concatenate((y, targets_memory))

        return x, y, memory_flags

    def get_custom_loader(
            self, class_indexes, memory=None, mode="test", data_source="train", sampler=None
    ):
        """
        Returns a custom loader.
            :param class_indexes: A list of class indexes that we want.
            :param mode: Various mode for the transformations applied on it.
            :param data_source: Whether to fetch from the train, val, or test set.
            :return: The raw data and a loader.
        """
        if not isinstance(class_indexes, list):  # TODO: deprecated, should always give a list
            class_indexes = [class_indexes]

        if data_source == "train":
            x, y = self.data_train, self.targets_train
        elif data_source == "val":
            x, y = self.data_val, self.targets_val
        elif data_source == "test":
            x, y = self.data_test, self.targets_test
        else:
            raise ValueError("Unknown data source <{}>.".format(data_source))

        data, targets = [], []
        for class_index in class_indexes:
            class_data, class_targets = self._select(
                x, y, low_range=class_index, high_range=class_index + 1
            )
            data.append(class_data)
            targets.append(class_targets)

        if len(data) == 0:
            assert memory is not None
        else:
            data = np.concatenate(data)
            targets = np.concatenate(targets)

        if (not isinstance(memory, tuple) and
            memory is not None) or (isinstance(memory, tuple) and memory[0] is not None):
            if len(data) > 0:
                data, targets, memory_flags = self._add_memory(data, targets, *memory)
            else:
                data, targets = memory
                memory_flags = np.ones((data.shape[0],))
        else:
            memory_flags = np.zeros((data.shape[0],))

        return data, self._get_loader(
            data, targets, memory_flags, shuffle=False, mode=mode, sampler=sampler
        )

    def get_memory_loader(self, data, targets):
        return self._get_loader(
            data, targets, np.ones((data.shape[0],)), shuffle=True, mode="train"
        )

    def _select(self, x, y, low_range=0, high_range=0):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _get_loader(self, x, y, memory_flags, shuffle=True, mode="train", sampler=None):
        if mode == "train":
            trsf = ds.transforms.c_transforms.Compose([*self.train_transforms, *self.common_transforms])
        elif mode == "test":
            trsf = ds.transforms.c_transforms.Compose([*self.test_transforms, *self.common_transforms])
        elif mode == "flip":
            trsf = ds.transforms.c_transforms.Compose([ds.vision.c_transforms.RandomHorizontalFlip(prob=1.),
                                                       *self.test_transforms, *self.common_transforms])
        else:
            raise NotImplementedError("Unknown mode {}.".format(mode))

        # TODO: Fix out what sampler do and its module is not completed
        if sampler is not None and mode == "train":
            logger.info("Using sampler {}".format(sampler))
            sampler = sampler(y, memory_flags, batch_size=self._batch_size, **self._sampler_config)
            batch_size = 1
        else:
            sampler = None
            batch_size = self._batch_size

        rank_id = 0
        rank_size = 1

        # rank_id = get_rank()
        # rank_size = get_group_size()

        gds = ds.GeneratorDataset(
            source=DummyDataset(x, y, memory_flags),
            column_names=["image", "label", "memory_flag"],
            num_parallel_workers=get_num_parallel_workers(self._workers),
            shuffle=shuffle if sampler is None else False, num_shards=rank_size, shard_id=rank_id,
            python_multiprocessing=False
        )
        gds = gds.map(operations=trsf, input_columns="image",
                      num_parallel_workers=get_num_parallel_workers(self._workers), )
        gds = gds.map(operations=ds.transforms.c_transforms.TypeCast(ms.float32), input_columns="image",
                      num_parallel_workers=get_num_parallel_workers(self._workers))
        gds = gds.map(operations=ds.transforms.c_transforms.TypeCast(ms.int32), input_columns="label",
                      num_parallel_workers=get_num_parallel_workers(self._workers))
        gds = gds.map(operations=ds.transforms.c_transforms.TypeCast(ms.int32), input_columns="memory_flag",
                      num_parallel_workers=get_num_parallel_workers(self._workers))
        gds = gds.batch(batch_size)  # TODO: whether to set drop remainder?
        return gds

    def _setup_data(
            self,
            datasets,
            random_order=False,
            class_order=None,
            seed=0,
            increment=10,
            validation_split=0.,
            initial_increment=None,
            data_path="data"
    ):
        self.data_train, self.targets_train = [], []
        self.data_test, self.targets_test = [], []
        self.data_val, self.targets_val = [], []
        self.increments = []
        self.class_order = []

        """
        The following maybe something supporting multi dataset, eg:CIFAR+ImageNet
        """
        current_class_idx = 0  # When using multiple datasets

        rank_id = 0
        rank_size = 1

        # rank_id = get_rank()
        # rank_size = get_group_size()

        for dataset in datasets:
            if dataset.__name__ == 'iCIFAR100':
                train_dataset = ds.Cifar100Dataset(data_path, usage='train',
                                                   num_parallel_workers=get_num_parallel_workers(self._workers),
                                                   num_shards=rank_size, shard_id=rank_id)
                test_dataset = ds.Cifar100Dataset(data_path, usage='test',
                                                  num_parallel_workers=get_num_parallel_workers(self._workers),
                                                  num_shards=rank_size, shard_id=rank_id)

                x_train, y_train = [], []

                for i, data in enumerate(train_dataset.create_dict_iterator()):
                    x_train.append(data['image'].asnumpy())
                    y_train.append(data['fine_label'].asnumpy())

                x_train = np.array(x_train)
                y_train = np.array(y_train)

                # print(y_train.shape)
                # print(y_train)

                x_val, y_val, x_train, y_train = self._split_per_class(
                    x_train, y_train, validation_split
                )

                x_test, y_test = [], []

                for i, data in enumerate(test_dataset.create_dict_iterator()):
                    x_test.append(data['image'].asnumpy())
                    y_test.append(data['fine_label'].asnumpy())

                x_test = np.array(x_test)
                y_test = np.array(y_test)

            # TODO: Maybe numworkers need to be set
            else:
                train_dataset = ds.ImageFolderDataset(os.path.join(data_path, 'train'),
                                                      num_parallel_workers=get_num_parallel_workers(self._workers),
                                                      num_shards=rank_size, shard_id=rank_id)
                test_dataset = ds.ImageFolderDataset(os.path.join(data_path, 'val'),
                                                     num_parallel_workers=get_num_parallel_workers(self._workers),
                                                     num_shards=rank_size, shard_id=rank_id)

                x_train, y_train = [], []

                for i, data in enumerate(train_dataset.create_dict_iterator()):
                    x_train.append(data['image'].asnumpy())
                    y_train.append(data['label'].asnumpy())

                x_train = np.array(x_train, dtype=object)
                y_train = np.array(y_train)

                x_val, y_val, x_train, y_train = self._split_per_class(
                    x_train, y_train, validation_split
                )

                x_test, y_test = [], []

                for i, data in enumerate(test_dataset.create_dict_iterator()):
                    x_test.append(data['image'].asnumpy())
                    y_test.append(data['label'].asnumpy())

                x_test = np.array(x_test, dtype=object)
                y_test = np.array(y_test)

            order = list(range(len(np.unique(y_train))))  # This is a list of 0 to max classnum, eg[0,1,...,99]
            if random_order:
                random.seed(seed)  # Ensure that following order is determined by seed:
                random.shuffle(order)
            elif class_order:
                order = class_order
            elif dataset.class_order is not None:
                order = dataset.class_order

            logger.info("Dataset {}: class ordering: {}.".format(dataset.__name__, order))

            self.class_order.append(order)

            # print(order)

            y_train = self._map_new_class_index(y_train, order)
            # y_val is always none
            y_val = self._map_new_class_index(y_val, order)
            y_test = self._map_new_class_index(y_test, order)

            y_train += current_class_idx
            y_val += current_class_idx
            y_test += current_class_idx

            current_class_idx += len(order)

            if len(datasets) > 1:
                self.increments.append(len(order))
            elif initial_increment is None:
                nb_steps = len(order) / increment
                remainder = len(order) - int(nb_steps) * increment

                if not nb_steps.is_integer():
                    logger.warning(
                        f"THe last step will have sligthly less sample ({remainder} vs {increment})."
                    )
                    self.increments = [increment for _ in range(int(nb_steps))]
                    self.increments.append(remainder)
                else:
                    self.increments = [increment for _ in range(int(nb_steps))]
            else:
                self.increments = [initial_increment]

                nb_steps = (len(order) - initial_increment) / increment
                remainder = (len(order) - initial_increment) - int(nb_steps) * increment
                if not nb_steps.is_integer():
                    logger.warning(
                        f"THe last step will have sligthly less sample ({remainder} vs {increment})."
                    )
                    self.increments.extend([increment for _ in range(int(nb_steps))])
                    self.increments.append(remainder)
                else:
                    self.increments.extend([increment for _ in range(int(nb_steps))])

            self.data_train.append(x_train)
            self.targets_train.append(y_train)
            self.data_val.append(x_val)
            self.targets_val.append(y_val)
            self.data_test.append(x_test)
            self.targets_test.append(y_test)

        self.data_train = np.concatenate(self.data_train)
        self.targets_train = np.concatenate(self.targets_train)
        self.data_val = np.concatenate(self.data_val)
        self.targets_val = np.concatenate(self.targets_val)
        self.data_test = np.concatenate(self.data_test)
        self.targets_test = np.concatenate(self.targets_test)

    @staticmethod
    def _map_new_class_index(y, order):
        """Transforms targets for new class order."""
        return np.array(list(map(lambda x: order.index(x), y)))

    @staticmethod
    def _split_per_class(x, y, validation_split=0.):
        """
        Splits train data for a subset of validation data.

        Split is done so that each class has a much data.

        But since PODNet's validation is 0.0, so we seems to have  x_val, y_val's shape[0] is 0
        eg: for cifar100 is (0,32,32,3)&&(0,), which means they are None
        """
        shuffled_indexes = np.random.permutation(x.shape[0])
        x = x[shuffled_indexes]
        y = y[shuffled_indexes]

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


class DummyDataset:
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

        # self.trsf = trsf

    def __len__(self):
        if isinstance(self.x, list):
            return len(self.x)
        else:
            return self.x.shape[0]

    def __getitem__(self, idx):
        x, y = self.x[idx], self.y[idx]
        memory_flag = self.memory_flags[idx]

        return x, y, memory_flag
        # if self.open_image:
        #     img = Image.open(x).convert("RGB")
        # else:
        #     img = Image.fromarray(x.astype("uint8"))

        # img = self.trsf(img)
        # return {"inputs": img, "targets": y, "memory_flags": memory_flag}


def _get_datasets(dataset_names):
    return [_get_dataset(dataset_name) for dataset_name in dataset_names.split("-")]


"""
return the class which define the data-process method defined in dataset.py 
"""


def _get_dataset(dataset_name):
    dataset_name = dataset_name.lower().strip()

    if dataset_name == "cifar100":
        return iCIFAR100
    elif dataset_name == "imagenet1000":
        return ImageNet1000
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


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


"""
Unit Test
"""

# a = _get_dataset("cifar100")
# print(a.__name__ == 'iCIFAR100')
# print(isinstance(a, iCIFAR100))
# x = IncrementalDataset(dataset_name='cifar100', data_path="D://file//Dataset//cifar-100-binary")
# The map_new_class_index func test
# def map_new_class_index(y, order):
#     """Transforms targets for new class order."""
#     return np.array(list(map(lambda x: order.index(x), y)))
# x = np.array([1, 4, 6, 8, 23])
# order = [4, 1, 8, 6, 23]
# print(map_new_class_index(x, order))  # [1 0 3 2 4]


# x = IncrementalDataset(
#     dataset_name='cifar100',
#     random_order=False,
#     shuffle=True,
#     batch_size=128,
#     workers=16,
#     validation_split=0.0,
#     onehot=False,
#     increment=1,
#     initial_increment=50,
#     sampler=None,
#     sampler_config={},
#     data_path='D://file//Dataset//cifar-100-binary',
#     class_order=
#     [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11,
#      4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82,
#      53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2,
#      95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39]
#
#     ,
#     seed=0,
#     dataset_transforms={'color_jitter': True},
#     all_test_classes=False,
#     metadata_path=None
# )

# order = [[87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11,
#           4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82,
#           53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85,
#           2,
#           95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39],
#          [58, 30, 93, 69, 21, 77, 3, 78, 12, 71, 65, 40, 16, 49, 89, 46, 24, 66, 19, 41, 5, 29, 15, 73, 11, 70, 90, 63,
#           67, 25, 59, 72, 80, 94, 54, 33, 18, 96, 2, 10, 43, 9, 57, 81, 76, 50, 32, 6, 37, 7, 68, 91, 88, 95, 85, 4, 60,
#           36, 22, 27, 39, 42, 34, 51, 55, 28, 53, 48, 38, 17, 83, 86, 56, 35, 45, 79, 99, 84, 97, 82, 98, 26, 47, 44,
#           62,
#           13, 31, 0, 75, 14, 52, 74, 8, 20, 1, 92, 87, 23, 64, 61],
#          [71, 54, 45, 32, 4, 8, 48, 66, 1, 91, 28, 82, 29, 22, 80, 27, 86, 23, 37, 47, 55, 9, 14, 68, 25, 96, 36, 90,
#           58,
#           21, 57, 81, 12, 26, 16, 89, 79, 49, 31, 38, 46, 20, 92, 88, 40, 39, 98, 94, 19, 95, 72, 24, 64, 18, 60, 50,
#           63,
#           61, 83, 76, 69, 35, 0, 52, 7, 65, 42, 73, 74, 30, 41, 3, 6, 53, 13, 56, 70, 77, 34, 97, 75, 2, 17, 93, 33, 84,
#           99, 51, 62, 87, 5, 15, 10, 78, 67, 44, 59, 85, 43, 11]]
