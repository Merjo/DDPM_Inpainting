import torch

print(torch.cuda.is_available())  # Test if GPU is available (important! should be TRUE!)
print(torch.cuda.current_device())
print(torch.cuda.get_device_name(torch.cuda.current_device()))
