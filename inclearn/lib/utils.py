import datetime
import warnings
import logging
import numpy as np
from mindspore.ops import stop_gradient
import mindspore.ops as ops

logger = logging.getLogger(__name__)


def check_loss(loss):
    is_nan = ops.IsNan()
    return not is_nan(loss) and loss >= 0.


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")
    # output: 20220512


def extract_features(model, loader):
    targets, features = [], []

    # set_train do influence the training para
    state = model.training
    model.set_train(mode=False)

    for i, data in enumerate(loader.create_dict_iterator()):
        inputs, _targets = data["image"], data["label"]

        _targets = _targets.asnumpy()
        # _features = model.extract(inputs.to(model.device)).detach().cpu().numpy()
        _features = model.extract(inputs).asnumpy()

        features.append(_features)
        targets.append(_targets)

    model.set_train(mode=state)

    return np.concatenate(features), np.concatenate(targets)


# TODO: need unit test
def add_new_weights(network, weight_generation, current_nb_classes, task_size, inc_dataset):
    print('22222222222222222222222222222222222222')
    if isinstance(weight_generation, str):
        warnings.warn("Use a dict for weight_generation instead of str", DeprecationWarning)
        weight_generation = {"type": weight_generation}

    if weight_generation["type"] == "imprinted":
        logger.info("Generating imprinted weights")

        network.add_imprinted_classes(
            list(range(current_nb_classes, current_nb_classes + task_size)), inc_dataset,
            **weight_generation
        )
    elif weight_generation["type"] == "basic":
        network.add_classes(task_size)

    else:
        raise ValueError("Unknown weight generation type {}.".format(weight_generation["type"]))

# print(get_date())
