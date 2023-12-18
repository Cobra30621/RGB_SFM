import torch
from utils import increment_path
from pathlib import Path
import shutil
import os

project = "paper experiment"
name = "SFMCNN"
group = "Rewrite"
tags = ["SFMCNN", "Rewrite"]
description = "12/18 老師架構圖實作"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

save_dir = increment_path('./runs/train/exp', exist_ok = False)
Path(save_dir).mkdir(parents=True, exist_ok=True)
print(save_dir)

lr_scheduler = {
    "name": "ReduceLROnPlateau",
    "args":{
        "patience": 100
    }
}

optimizer = {
    "name":"Adam",
    "args":{

    }
}

arch = {
    "name": 'SFMCNN',
    "args":{
        "in_channels": 3,
        "out_channels": 15,
        "Conv2d_kernel": [(1, 1), (1, 1), (1, 1), (1, 1)],
        "SFM_filters": [(2, 2), (2, 2), (1, 3)],
        "channels": [1, 16, 32, 64, 128],
        "strides": [1, 1, 1, 1],
        "paddings": [3, 3, 0, 0],
        "w_arr": [2.0, 100.0, 784.0, 32.0],
        # "w_arr": [9.0, 400.0, 1568.0, 64.0],
        # "w_arr": [4.5, 200.0, 784.0, 32.0],
        "percent": [0.2, 0.3, 0.5, 0.5],
        "fc_input": 3*1*3*128,
        "device": device
    }
}

# arch = {
#     "name": 'CNN',
#     "args":{
#         "in_channels": 3,
#         "out_channels": 15,
#     }
# }

# arch = {
#     "name": 'AlexNet',
#     "args":{
#         "num_classes":15
#     }
# }

config = {
    "device": device,
    "root": os.path.dirname(__file__),
    "save_dir": save_dir,
    "model": arch,
    "dataset": 'rgb_simple_shape',
    "input_shape": (28, 28),
    "rbf": "triangle",
    "batch_size": 32,
    "epoch" : 200,
    "lr" : 0.001,
    "lr_scheduler": lr_scheduler,
    "optimizer": optimizer,
    "loss_fn": "CrossEntropyLoss",
}