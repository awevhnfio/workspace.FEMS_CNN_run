# import torch
from tensorflow.python.client import device_lib

print(device_lib.list_local_devices())

exit()

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

