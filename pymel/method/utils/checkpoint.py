from typing import Any
from typing import *
import numpy as np
import torch
from torch import nn


class ModelCheckPoint:
    def __init__(self, 
                 save_dir:str = None,
                 extension:str = "pt",
                 save_last: bool = False,
                 save_best: bool = True) -> None:
        
        if save_dir is None:
            raise ValueError(f"save_dir cannot be None")
        elif not isinstance(save_dir, str):
            raise TypeError(f"save_dir must be a string, \
                but found {type(save_dir)} instead")
        else:
            self.sv = save_dir
            
        if extension is None:
            raise ValueError(f"extension cannot be None")
        elif not isinstance(extension, str):
            raise TypeError(f"extension must be a string, \
                but found {type(extension)} instead")
        elif extension not in ["pt", "pth"]:
            raise ValueError(f"extention must be 'pt' or 'pth', \
                but found {extension} instead")
        else:
            self.ext = extension
            
        for saven, save_type in zip(["save_last", "save_best"], [save_last, save_best]):
            if not isinstance(save_type, bool):
                raise TypeError(f"{saven} must be a boolean, \
                    but found {type(save_type)} instead")
        
        self.sv_best = save_best
        self.sv_last = save_last
        
        self.old_loss = 1e26
        self.old_acc = 0
    
    def __call__(self, 
                 model: nn.Module, 
                 loss: list[float] or Dict[str, float] = None, 
                 acc: list[float] or Dict[str, float] = None, 
                 optimizer: torch.optim = None, 
                 epoch: int = None,
                 *args: Any, **kwds: Any) -> Any:
        if not isinstance(loss, list):
            raise TypeError(f"loss must be a list, but found {type(loss)}, instead")
        if not isinstance(acc, list):
            raise TypeError(f"acc must be a list, but found {type(acc)}, instead")
        
        for metric_name, metric_data in zip(["loss", "acc"], [loss, acc]):
            if metric_data is not None:
                metric_val_lst = list(metric_data.values) if isinstance(metric_data, dict) else metric_data
                if np.sum(
                    np.array([
                        isinstance(x, int) or isinstance(x, float) or isinstance(x) 
                        for x in metric_val_lst
                    ])
                ) > 0:
                    raise ValueError(f"There are some elements that is not support type\
                        (int, float) in metric {metric_name} list")
                    
        loss_val_lst = list(loss.values) if isinstance(loss, dict) else loss
        acc_val_lst = list(acc.values) if isinstance(acc, dict) else acc
        mean_loss = np.mean(np.array(loss_val_lst)).item()
        mean_acc = np.mean(np.array(acc_val_lst)).item()
        
        if self.sv_best:
            if mean_loss < self.old_loss and mean_acc < self.old_acc:
                self.old_loss = mean_loss
                self.old_acc = mean_acc
            
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                    'acc' : acc
                }, self.sv + f"/best.{self.ext}")
        if self.sv_last:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'acc' : acc
            }, self.sv + f"/best.{self.ext}")