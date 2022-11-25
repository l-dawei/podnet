import sys
import os

# npwd = os.getcwd()
# sys.path.append(npwd)
# path1 = '/home/work/user-job-dir/code'
# sys.path.append("/home/work/user-job-dir/code")
# os.system('pip install scikit-learn')
# sys.path.append("/cache/code/podnet")

from inclearn import parser
from inclearn.train import train
from mindspore import context
from mindspore.communication import init
from mindspore.context import ParallelMode
import mindspore as ms

# import moxing as mox

"""
The main function,  and the entrance of this project
"""


def main():
    args = parser.get_parser().parse_args()

    """
        将数据集从obs拷贝到训练镜像中
        """
    # obs_data_url = args.data_url
    # args.data_url = '/home/work/user-job-dir/inputs/data/'
    # obs_train_url = args.train_url
    # args.train_url = '/home/work/user-job-dir/outputs/model/'
    # try:
    #     mox.file.copy_parallel(obs_data_url, args.data_url)
    #     print("Successfully Download {} to {}".format(obs_data_url,
    #                                                   args.data_url))
    # except Exception as e:
    #     print('moxing download {} to {} failed: '.format(
    #         obs_data_url, args.data_url) + str(e))

    """
    将数据集从obs拷贝到训练镜像中
    """

    # 将dataset_path指向data_url
    # args.dataset_path = args.data_url

    args = vars(args)  # Converting argparse Namespace to a dict.

    if args["seed_range"] is not None:
        args["seed"] = list(range(args["seed_range"][0], args["seed_range"][1] + 1))
        print("Seed range", args["seed"])

    """
    Set Context
    """
    # context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    # context.set_context(mode=context.PYNATIVE_MODE, device_target='Ascend', device_id=1)
    context.set_context(mode=context.GRAPH_MODE, device_target='Ascend', device_id=2)
    # context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=int(os.environ["DEVICE_ID"]))
    # context.set_auto_parallel_context(parallel_mode=ParallelMode.DATA_PARALLEL)
    # init()

    """ 
    PODNet Cifar100
    """
    # path = "/home/work/user-job-dir/code//options//podnet//podnet_cnn_cifar100.yaml", "/home//work//user-job-dir/code//options//data//cifar100_3orders.yaml"

    path = "options//podnet//podnet_cnn_cifar100.yaml", "options//data//cifar100_3orders.yaml"

    args['options'] = path
    # args['data_path'] = 'D://file//Dataset//cifar-100-binary'
    # args['data_path'] = '/disk0/dataset/cifar-100-binary'
    args['data_path'] = '/data1/dataset/cifar-100-binary'
    # args['data_path'] = os.path.join(args['data_url'], 'cifar-100-binary')
    # '/home/work/user-job-dir/inputs/data/'

    args['initial_increment'] = 50
    args['increment'] = 1
    args['fixed_memory'] = True
    args['label'] = 'podnet_cnn_cifar100_50steps'

    args['workers'] = 32
    # args['workers'] = 1

    # print(args)

    for _ in train(args):  # `train` is a generator in order to be used with hyper find.
        pass


if __name__ == "__main__":
    main()
