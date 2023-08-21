import os, sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
from typing import *
from config import DSConfig, TrainConfig
from utils import ModelCheckPoint
import torch
from torch import nn
from torch.optim import *
import numpy as np

opt_mapping = {
    'adam' : Adam
}

class Trainer:
    def __init__(self, ds_cfg: DSConfig, tr_cfg: TrainConfig,
                 model: nn.Module = None, gpus: List[int] = [0]) -> None:
        # if not isinstance(ds_cfg, DSConfig):
        #     raise TypeError(f"PyMel GPT: ds_cfg must be DSConfig, but found {type(ds_cfg)} instead")
        # else:
        #     self.ds_cfg = ds_cfg
            
        # if not isinstance(tr_cfg, TrainConfig):
        #     raise TypeError(f"PyMel GPT: tr_cfg must be TrainConfig, but found {type(tr_cfg)} instead")
        # else:
        #     self.tr_cfg = tr_cfg
        
        self.ds_cfg = ds_cfg
        self.tr_cfg = tr_cfg
        
        if self.tr_cfg.checkpoint():
            self.checker = ModelCheckPoint(
                save_dir = self.tr_cfg.get_sv_dir(),
                save_best = self.tr_cfg.get_sv_best(),
                save_last = self.tr_cfg.get_sv_last(),
                extension = self.tr_cfg.get_ext()
            )
        
        if model is None:
            raise ValueError("PyMel GPT: model cannot be None")    
        elif not isinstance(model, nn.Module):
            raise TypeError(f"PyMel GPT: model must be a nn.Module type, \
                but found {type(model)} instead")
        else:
            self.model = model
            
        if len(gpus) > torch.cuda.device_count():
            raise ValueError(f"PyMel GPT: The number of availabel GPUs: \
                {torch.cuda.is_available()} but found required {len(gpus)}\
                    in gpus: {gpus}")
        elif np.sum(
            np.array(
                [not isinstance(x, int) for x in gpus]
            )
        ) > 0:
            raise ValueError(f"PyMel GPT: the elements inside gpus must be integers")
        else:
            self.gpus = gpus            
    
    def train(self):
        raise NotImplementedError()
    
    def main_worker(self):
        raise NotImplementedError()
    
    def config_export(self):
        return {
            "training_config" : self.tr_cfg.config_export(),
            "dataset_config" : self.ds_cfg.config_export(),
            "gpus" : self.gpus
        }