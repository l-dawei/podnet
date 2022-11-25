import torch

x=torch.tensor([1,2,3])
output=torch.frobenius_norm(x)
print(output)