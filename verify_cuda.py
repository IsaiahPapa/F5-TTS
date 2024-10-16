import torch

# Check if CUDA is available
print(f"CUDA available: {torch.cuda.is_available()}")

# Check CUDA version
print(f"CUDA version: {torch.version.cuda}")

# Check the number of CUDA devices
print(f"Number of CUDA devices: {torch.cuda.device_count()}")

# Get the name of the current CUDA device
if torch.cuda.is_available():
    print(f"Current CUDA device: {torch.cuda.get_device_name(0)}")

# Check if your tensors are on CUDA
x = torch.rand(5, 3)
print(f"Is tensor on CUDA? {x.is_cuda}")

# Move a tensor to CUDA and check again
if torch.cuda.is_available():
    x = x.cuda()
    print(f"Is tensor on CUDA now? {x.is_cuda}")