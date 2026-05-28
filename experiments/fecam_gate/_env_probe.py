"""Quick env probe for Jett's FeCAM γ measurement.
Prints torch / torchvision / timm availability and CUDA status.
"""
import sys
print("python", sys.version)
try:
    import torch
    print("torch", torch.__version__)
    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    if torch.cuda.is_available():
        print("device_name", torch.cuda.get_device_name(0))
except Exception as e:
    print("torch IMPORT FAIL:", e)
try:
    import torchvision
    print("torchvision", torchvision.__version__)
except Exception as e:
    print("torchvision IMPORT FAIL:", e)
try:
    import timm
    print("timm", timm.__version__)
except Exception as e:
    print("timm IMPORT FAIL:", e)
