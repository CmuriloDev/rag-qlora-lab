import torch

print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

x = torch.randn(1000,1000).cuda()
y = x @ x

print(y.shape)