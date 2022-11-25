import mindspore.dataset as ds
from PIL import Image
import mindspore.ops as ops

dataset_path = "D://file//Dataset//cifar-100-binary"
dataset = ds.Cifar100Dataset(dataset_path, usage='train')
# print(ci.get_dataset_size()) # train: 50000

# i is from 0 to 5e4-1
for i, data in enumerate(dataset.create_dict_iterator()):
    # print(len(data)) 3
    # print("Image shape: {}".format(data['image'].shape), ", Fine_Label {}".format(data['fine_label']),
    #       ", Coarse_Label {}".format(data['coarse_label']))

    images = data['image']
    images = images.asnumpy()  # mindspore.Tensor to numpy
    # print(images.dtype)   uint8
    # print(images.shape)   (32, 32, 3)

    finelabel = data['fine_label']
    print(finelabel.asnumpy())  # mindspore tensor of int

    if i >= 10:
        break
