from mindspore import context
import mindspore.dataset as ds
import mindspore.dataset.vision
from mindspore.dataset.transforms.py_transforms import Compose
from testmodule.basic.model import LeNet5
import mindspore.nn as nn
from mindspore import Model
from mindspore.train.callback import LossMonitor


def main():
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')

    train_path = "D://file//Dataset//flower//train"
    test_path = "D://file//Dataset//flower//test"
    data_path = "D://file//Dataset//flower//flower_photos//daisy//5547758_eea9edfd54_n.jpg"

    # img = Image.open(data_path)
    # print(ds.vision.py_transforms.ToTensor(img))

    data_transform = {
        "train": Compose([ds.vision.py_transforms.Decode(),
                            ds.vision.c_transforms.RandomResizedCrop(224),
                          ds.vision.c_transforms.RandomHorizontalFlip(),
                          ds.vision.py_transforms.ToTensor(),
                          ds.vision.c_transforms.HWC2CHW(),
                          ds.vision.c_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                          ])
    }
    train_ds = ds.ImageFolderDataset(train_path)
    train_ds = train_ds.map(operations=data_transform["train"], input_columns="image")
    train_ds = train_ds.batch(32, drop_remainder=True)

    net = LeNet5()
    num_epochs = 5

    opt = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)

    loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    model = Model(net, loss, opt, metrics={"Accuracy": nn.Accuracy()})

    eval_param_dict = {"model": model, "dataset": train_ds, "metrics_name": "Accuracy"}

    from mindspore.train.callback import ModelCheckpoint, CheckpointConfig
    # 设置模型保存参数
    config_ck = CheckpointConfig(save_checkpoint_steps=1875, keep_checkpoint_max=10)
    # 应用模型保存参数
    ckpoint = ModelCheckpoint(prefix="checkpoint_lenet", config=config_ck)

    # eval_cb = EvalCallBack(apply_eval, eval_param_dict, )

    model.train(num_epochs, train_ds, callbacks=[ckpoint, LossMonitor(125)], dataset_sink_mode=False)

    # print(train_ds)


if __name__ == '__main__':
    main()
