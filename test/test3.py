import mindspore.dataset as ds

# def create_dataset(data_path, batch_size=32, repeat_size=1,
#                    num_parallel_workers=1):

dataset_path = "D://file//Dataset//cifar-100-binary"

from PIL import Image
import matplotlib.pyplot as plt

sampler = ds.SequentialSampler(num_samples=6)
dataset = ds.Cifar100Dataset(dataset_path, sampler=sampler)

# 在数据集上创建迭代器,检索到的数据将是字典数据类型
for i, data in enumerate(dataset.create_dict_iterator()):
    # print("Image shape: {}".format(data['image'].shape), ", Label {}".format(data['label']))
    print(len(data))
    # print("Image shape: {}".format(data['image'].shape), ", Fine_Label {}".format(data['fine_label']), ", Coarse_Label {}".format(data['coarse_label']))
    image = data['image']
    image = image.asnumpy()  # mindspore.Tensor to numpy
    image = Image.fromarray(image)
    # plt
    plt.subplot(2, 3, i + 1)
    plt.imshow(image)
    plt.title(f"{i + 1}", fontsize=6)
    plt.xticks([])
    plt.yticks([])

plt.show()
