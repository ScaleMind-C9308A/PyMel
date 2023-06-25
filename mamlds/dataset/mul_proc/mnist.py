import os, sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
import torch
from typing import Callable, Optional, Any, Tuple
from torchvision.datasets import MNIST
import random
from torch.utils.data import DataLoader
from core import MUL_PROC_MAML_DATASET

class MamlMnistV2(MNIST, MUL_PROC_MAML_DATASET):
    def __init__(self, 
                 root: str = "~/data", 
                 train: bool = True, 
                 transform: Callable[..., Any] = None, 
                 target_transform: Callable[..., Any] = None, 
                 download: bool = False,
                 k_shot: int = 5, 
                 k_query: int = 5,
                 n_train_cls:int = -1,
                 merge_task:bool = True                             
                 ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        
        self.check_int_arg(k_shot=k_shot, k_query=k_query, n_train_cls=n_train_cls)
        self.check_bool_arg(merge_task=merge_task)
        
        self.ks = k_shot
        self.kq = k_query
        self.nt = len(self.class_to_idx)
        self.mt = merge_task
        
        self.__setup__()   
    
    def __setup__(self):
        self.dict_ds = {
            _cls : [] for _cls in list(self.class_to_idx.values())
        }        

        for idx, _cls in enumerate(self.targets):
            self.dict_ds[_cls.item()].append(self.data[idx])
        
        self.max_sample = max([len(self.dict_ds[_cls]) for _cls in self.dict_ds])
        self.sample_cls_cnt = self.max_sample + self.max_sample % (self.ks + self.kq)
        
        sampled_dict_ds = {
            _cls : self.dict_ds[_cls] + random.sample(
                self.dict_ds[_cls], k = self.sample_cls_cnt - len(self.dict_ds[_cls])
            ) for _cls in list(self.dict_ds.keys())
        }
                
        self.dict_ds = sampled_dict_ds
    
    def __len__(self) -> int:
        return self.sample_cls_cnt
    
    def __getitem__(self, index: int):
        
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
        
        
# """
if __name__ == "__main__":
    ds = MamlMnistV2(train=False, download=True)
    
    print(f"len dataset: {len(ds)}, sample per class: {ds.sample_cls_cnt}")        
    
    # Check compatibility with DataLoader
    dl = DataLoader(
        dataset=ds,
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )  
    
    print(f"Len Data Loader: {len(dl)}")
    
    for idx, data in enumerate(dl):
        for task in data:
            print(f"Task: {task} - Count: {len(data[task])} - Shape: {data[task].shape}")

# """