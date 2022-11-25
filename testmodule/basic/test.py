import os
import argparse
from mindspore import context
import mindspore.dataset as ds
import mindspore.dataset.transforms.c_transforms as C
import mindspore.dataset.vision.c_transforms as CV
from mindspore.dataset.vision import Inter
from mindspore import dtype as mstype
import mindspore.nn as nn
from model import LeNet5
from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
from mindspore.nn import Accuracy
from mindspore.train.callback import LossMonitor
from mindspore import Model


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    # 定义数据集
    mnist_ds = ds.MnistDataset(data_path)
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # 定义所需要操作的map映射
    trans = [
        CV.Resize((resize_height, resize_width), interpolation=Inter.LINEAR),
        CV.Rescale(rescale, shift),
        CV.Rescale(rescale_nml, shift_nml),
        CV.HWC2CHW()
    ]
    # trans = C.Compose(trans)
    # print(type(trans))

    type_cast_op = C.TypeCast(mstype.int32)
    # 使用map映射函数，将数据操作应用到数据集
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=trans, input_columns="image",
                            num_parallel_workers=num_parallel_workers)

    # 进行shuffle、batch、repeat操作
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(count=repeat_size)

    return mnist_ds


def train_net(model, epoch_size, data_path, repeat_size, ckpoint_cb, sink_mode):
    """定义训练的方法"""
    # 加载训练数据集
    ds_train = create_dataset(os.path.join(data_path, "train"), 32, repeat_size)
    model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(125)], dataset_sink_mode=sink_mode)


def main():
    # parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    # 实例化网络
    net = LeNet5()

    # 定义损失函数
    net_loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # 定义优化器
    net_opt = nn.Momentum(net.trainable_params(), learning_rate=0.01, momentum=0.9)

    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)

    # 应用模型保存参数
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck, directory='./checkpoint')

    train_epoch = 4
    mnist_path = "D://file//Dataset//Mnist"
    dataset_size = 1
    model = Model(net, net_loss, net_opt, metrics={"Accuracy": Accuracy()})
    train_net(model, train_epoch, mnist_path, dataset_size, ckpoint, False)
    # test_net(model, mnist_path)


if __name__ == "__main__":
    main()
