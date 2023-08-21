import os, sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
import argparse
import copy
import random

from torchvision import transforms
from torch import nn

from pymel.config import DSConfig, TrainConfig
from pymel.base_model import CNN_Mnist
from pymel.method.gradient_based import FSMAML

if __name__ == "__main__":
    
    k_shot = 5
    k_query = 5
    outer_epoch = 10
    inner_epoch = 1
    meta_opt = sp_opt = 'adam'
    meta_wd = sp_wd = 1e-04
    meta_lr = 0.001
    sp_lr = 0.01
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    ds_conf = DSConfig(
        dataset="mnist",
        download=True,
        transform=transform,
        k_shot=k_shot,
        k_query=k_query,
        num_worker=os.cpu_count(),
        pin_memory=True
    )
    
    tr_conf = TrainConfig(
        checkpoint=True,
        logging=True,
        save_best=True,
        save_last=True,
        extension="pt"
    )
    
    trainer = FSMAML(
        ds_cfg=ds_conf,
        tr_cfg=tr_conf,
        model=CNN_Mnist((1, 28, 28), 10),
        gpus=[0],
        meta_opt=meta_opt,
        meta_lr=meta_lr,
        meta_wd=meta_wd,
        sp_opt=sp_opt,
        sp_lr=sp_lr,
        sp_wd=sp_wd,
        criterion=nn.CrossEntropyLoss(),
        outer_epoch=outer_epoch,
        inner_epoch=inner_epoch
    )
    
    trainer.train()