import torch

if torch.cuda.is_available():
    print("CUDA is available. PyTorch can use the GPU!")
    print(f"Using device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Using the CPU.")