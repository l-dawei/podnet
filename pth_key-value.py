import torch

def traversal_params(pth_file_path, ckpt_file_path):
    # load pth file as a dictionary
    torch_params_dict = torch.load(pth_file_path)
    # traversal a params dictionary
    for k, v in torch_params_dict.items():
        print("param_key: ", k)
        print("param_value: ", v)