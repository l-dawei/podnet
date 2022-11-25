from inclearn.lib import factory, data, network
from inclearn.lib.data import IncrementalDataset
import mindspore.nn as nn
from mindspore import context

# scheduler = nn.CosineDecayLR(min_lr=.0, max_lr=0.1, decay_steps=160)

"""
context
"""

context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')

"""
ds part
"""
inc_dataset = IncrementalDataset(
    dataset_name='cifar100',
    random_order=False,
    shuffle=True,
    batch_size=128,
    workers=1,
    validation_split=0.0,
    onehot=False,
    increment=1,
    initial_increment=50,
    sampler=None,
    sampler_config={},
    data_path='D://file//Dataset//cifar-100-binary',
    class_order=
    [87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86, 84, 96, 18, 24, 32, 45, 88, 11,
     4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83, 17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82,
     53, 9, 31, 75, 38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76, 40, 30, 23, 85, 2,
     95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39],
    seed=0,
    dataset_transforms={'color_jitter': True},
    all_test_classes=False,
    metadata_path=None
)

memory, memory_val = None, None
task_info, train_loader, val_loader, test_loader = inc_dataset.new_task(memory, memory_val)

# print(inc_dataset)
"""
backbone part
"""
_network = network.BasicNet(
    'rebuffi',
    convnet_kwargs={},
    classifier_kwargs={'type': 'cosine', 'proxy_per_class': 10, 'distance': 'neg_stable_cosine_distance'},
    postprocessor_kwargs={'type': 'learned_scaling', 'initial_value': 1.0},
    device=[0],
    return_features=True,
    extract_no_act=True,
    classifier_no_act=True,
    attention_hook=True,
    gradcam_hook=False
)
yy = _network.classifier
print(yy)
dict_datasets = next(train_loader.create_dict_iterator())
output = _network(dict_datasets['image'])
print(output)
