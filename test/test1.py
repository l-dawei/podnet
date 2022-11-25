import mindspore
import numpy as np
from mindspore import Tensor
import mindspore.ops as ops
import mindspore.context as context
import os
import logging

# def test_logging():
#     logging.debug('Python debug')
#     logging.info('Python info')
#     logging.warning('Python warning')
#     logging.error('Python Error')
#     logging.critical('Python critical')


import os
import requests

requests.packages.urllib3.disable_warnings()


def download_dataset(dataset_url, path):
    filename = dataset_url.split("/")[-1]
    save_path = os.path.join(path, filename)
    if os.path.exists(save_path):
        return
    if not os.path.exists(path):
        os.makedirs(path)
    res = requests.get(dataset_url, stream=True, verify=False)
    with open(save_path, "wb") as f:
        for chunk in res.iter_content(chunk_size=512):
            if chunk:
                f.write(chunk)
    print("The {} file is downloaded and saved in the path {} after processing".format(os.path.basename(dataset_url),
                                                                                       path))


if __name__ == '__main__':
    # context.set_context(device_target="Ascend")
    x = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
    # y = Tensor(np.ones([1, 3, 3, 4]).astype(np.float32))
    # print(ops.add(x, y))
    # test_logging()
    # print(os.environ["CUDA_VISIBLE_DEVICES"])
    # path = os.path.join('.', f"net_{2222}_task_{'ssssss'}.ckpt")
    # train_path = "D://file//Dataset//Cifar10//train"
    # test_path = "D://file//Dataset//Cifar10//test"
    #
    # download_dataset("https://mindspore-website.obs.myhuaweicloud.com/notebook/datasets/mnist/t10k-images-idx3-ubyte",
    #                  test_path)
    # print(path)
    # x.view(3, 4, -1)
    # print(x.shape)

    a = np.ones((1000, 10))
    b = np.ones(1000)
    xx = Tensor(a, mindspore.float32)
    yy = Tensor(b, mindspore.float32)
    # c = xx.shape
    # print(type(xx))
    # e = np.array([1, 4, 7, 2, 3, 5, 1, 2])
    # e = Tensor(e)
    # ds = np.unique(e)
    topops = ops.TopK(True)
    _, c = topops(xx, 1)
    print(c.shape)
    c = c.T
    print(c.shape)
    # aa = Tensor(([1, 2, 4], [7, 6, 9])).astype(mindspore.float32)
    # print(type(aa.T))

    t1 = yy.view(1, -1)
    print(t1.shape)
    zz = yy.view(1, -1).expand_as(c)
    print(zz.shape)
    reshape = ops.Reshape()
    zz = reshape(zz, (-1,))
    print(zz.shape)

    y = Tensor([1.0])
    x = float(y.asnumpy()[0])
    print(type(x))
    print(type(y.item()))
