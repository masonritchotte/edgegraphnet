# Check if CUDA is available
import torch, os

if torch.cuda.is_available():
    print("CUDA is available")
else:
    print("CUDA is not available")

# Check if other process using cuda
if torch.cuda.current_device() != 0:
    print("Other process is using CUDA")