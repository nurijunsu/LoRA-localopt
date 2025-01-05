import torch, torchvision
print(torchvision.__version__)         # PyTorch version
print(torch.version.cuda)        # CUDA version used by PyTorch
print(torch.cuda.is_available()) # Check if CUDA is available
