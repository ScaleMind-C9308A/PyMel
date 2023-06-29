import os, sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
import torch
from typing import Callable, Optional, Any, Tuple
from torchvision.datasets import MNIST
from PIL import Image
import random
from torch.utils.data import DataLoader
from .core import MUL_PROC_MAML_DATASET

class MamlMnist(MNIST, MUL_PROC_MAML_DATASET):
    def __init__(self, 
                 root: str = "~/data", 
                 train: bool = True, 
                 transform: Callable[..., Any] = None, 
                 target_transform: Callable[..., Any] = None, 
                 download: bool = False,
                 k_shot: int = 5, 
                 k_query: int = 5,
                 n_train_cls:int = -1,
                 merge_task:bool = True,
                 maml: bool = True                             
                 ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        
        self.check_int_arg(k_shot=k_shot, k_query=k_query, n_train_cls=n_train_cls)
        self.check_bool_arg(merge_task=merge_task, maml = maml)
        
        self.ks = k_shot
        self.kq = k_query
        self.nt = len(self.class_to_idx)
        self.mt = merge_task
        self.maml = maml
        
        if maml:
            self.__setup__()   
    
    def __setup__(self):
        self.dict_ds = {
            int(_cls) : [] for _cls in list(self.class_to_idx.values())
        }        

        for idx, _cls in enumerate(self.targets):
            self.dict_ds[_cls.item()].append(
                Image.fromarray(self.data[idx].numpy(), mode="L")
            )
        
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

if __name__ == "__main__":
    from torchvision import transforms
    ds = MamlMnist(train=True, download=True, transform=transforms.ToTensor(),)
    
    print(ds.dict_ds.keys())
    
    for task in ds.dict_ds:
        print(len(ds.dict_ds[task]))
        
    dl = DataLoader(ds, batch_size=30)
    
    for idx, batch in enumerate(dl):
        
        tasks = random.sample(list(batch.keys()), k=10)
        print(tasks)
        
        for task in tasks:
            print(f"Task: {task} - {len(batch[task])}")