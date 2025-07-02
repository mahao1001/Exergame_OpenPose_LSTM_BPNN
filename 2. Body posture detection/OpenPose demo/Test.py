import torch, platform
print("Python :", platform.python_version())
print("Torch   :", torch.__version__)
print("Wheel   :", torch.version.cuda)          # 应显示 12.1
print("GPU OK? :", torch.cuda.is_available())   # True 才算成功
if torch.cuda.is_available():
    print("Device :", torch.cuda.get_device_name(0))

