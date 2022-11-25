import mindspore.nn as nn
from inclearn.convnet import my_resnet, resnet
from inclearn import models
from inclearn.lib import data
import warnings

"""
It seems that PODNet_CNN_CIFAR100/ImageNet1000 only use sgd
"""

# def get_optimizer(params, optimizer, lr, weight_decay=0.0):
#     if optimizer == "sgd":
#         return nn.SGD(params, learning_rate=lr, weight_decay=weight_decay, momentum=0.9)
# elif optimizer == "adam":
#     return nn.Adam(params, lr=lr, weight_decay=weight_decay)
# elif optimizer == "adamw":
#     return nn.AdamWeightDecay(params, lr=lr, weight_decay=weight_decay)
# elif optimizer == "sgd_nesterov":
#     return nn.SGD(params, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)


# raise NotImplementedError

"""
rebuffi for podnet-cifar100
resnet18 for podnet-imagenet1k
"""


def get_convnet(convnet_type, **kwargs):
    if convnet_type == "resnet18":
        return resnet.resnet18(**kwargs)
    if convnet_type == "rebuffi":
        return my_resnet.resnet_rebuffi(**kwargs)


def get_model(args):
    dict_models = {
        "podnet": models.PODNet
    }

    model = args["model"].lower()

    if model not in dict_models:
        raise NotImplementedError(
            "Unknown model {}, must be among {}.".format(args["model"], list(dict_models.keys()))
        )

    return dict_models[model](args)


def get_data(args, class_order=None):
    return data.IncrementalDataset(
        dataset_name=args["dataset"],
        random_order=args["random_classes"],
        shuffle=True,
        batch_size=args["batch_size"],
        workers=args["workers"],
        validation_split=args["validation"],
        onehot=args["onehot"],
        increment=args["increment"],
        initial_increment=args["initial_increment"],
        # sampler=get_sampler(args),
        sampler=None,
        sampler_config=args.get("sampler_config", {}),
        data_path=args["data_path"],
        class_order=class_order,
        seed=args["seed"],
        dataset_transforms=args.get("dataset_transforms", {}),
        all_test_classes=args.get("all_test_classes", False),
        metadata_path=args.get("metadata_path")
    )


"""
need to convert to mindspore
"""


# def set_device(args):


# def get_sampler(args):
#     if args["sampler"] is None:
#         return None

# def get_lr_scheduler(
#         scheduling_config, nb_epochs
# ):
#     if scheduling_config is None:
#         return None
#
#     # TODO: lr scheduler
#     if scheduling_config == "cosine":
#         scheduler = nn.CosineDecayLR(min_lr=.0, max_lr=0.1, decay_steps=nb_epochs)
#     else:
#         raise ValueError("Unknown LR scheduling type {}.".format(scheduling_config["type"]))
#
#     return scheduler
