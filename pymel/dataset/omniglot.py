import os, sys
from typing import Callable, Optional
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
import numpy as np
import torch
from typing import *
from torchvision.datasets import Omniglot
from PIL import Image
import random
from torch.utils.data import DataLoader
from .core import MUL_PROC_MAML_DATASET

class MamlOmniglot(Omniglot, MUL_PROC_MAML_DATASET):
    def __init__(self, 
                 root: str, 
                 background: bool = True, 
                 transform: Callable[..., Any] or None = None, 
                 target_transform: Callable[..., Any] or None = None, 
                 download: bool = False,
                 k_shot: int = 5, 
                 k_query: int = 5,
                 n_train_cls:int = -1,
                 merge_task:bool = True,
                 maml: bool = True) -> None:
        super().__init__(root, background, transform, target_transform, download)
    
        self.check_int_arg(k_shot=k_shot, k_query=k_query, n_train_cls=n_train_cls)
        self.check_bool_arg(merge_task=merge_task, maml = maml)
        
        self.ks = k_shot
        self.kq = k_query
        self.nt = len(self.class_to_idx)
        self.mt = merge_task
        self.maml = maml
    
    def __setup__(self):
        
        int_cls = np.unique(
            np.array(
                [character_class for _, character_class in self._flat_character_images]
            )
        ).tolist()
        
        self.dict_ds = {
            int(character_class) : [] for character_class in int_cls
        }
        
        for index in range(len(self._flat_character_images)):
            image_name, character_class = self._flat_character_images[index]
            
            image_path = os.path.join(
                self.target_folder, self._characters[character_class], image_name
            )
            image = Image.open(image_path, mode="r").convert("L")
            
            self.dict_ds[int(character_class)].append(image)
            
        self.max_sample = max([len(self.dict_ds[_cls]) for _cls in self.dict_ds])
        self.sample_cls_cnt = self.max_sample + ((self.ks + self.kq) - self.max_sample % (self.ks + self.kq))
        
        sampled_dict_ds = {
            _cls : self.dict_ds[_cls] + random.sample(
                self.dict_ds[_cls], k = self.sample_cls_cnt - len(self.dict_ds[_cls])
            ) for _cls in list(self.dict_ds.keys())
        }
        
        self.dict_ds = sampled_dict_ds
    
    def __len__(self) -> int:
        if self.maml:
            return self.sample_cls_cnt
        else:
            return super().__len__()
    
    def __getitem__(self, index: int):
        if self.maml:
            if index >= len(self):
                raise ValueError("Data set index of out range")
            elif index == -1:
                index = len(self) - 1      
            
            selected_dict = {
                _cls : self.transform(
                    self.dict_ds[_cls][index]
                ) if self.transform is not None else self.dict_ds[_cls][index] for _cls in range(self.nt)
            }
            
            return selected_dict
        else:
            return super().__getitem__(index=index)