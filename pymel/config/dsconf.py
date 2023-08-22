import os, sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
from typing import *
from dataset import *

ds_map = {
    "mnist" : MamlMnist,
    "kmnist" : MamlKMnist
}

class DSConfig:
    def __init__(self, 
                 dataset: str = None, 
                 root: str = "~/data",
                 download: bool = True,
                 transform: Callable[..., Any] = None, 
                 target_transform: Callable[..., Any] = None, 
                 k_shot: int = 5, 
                 k_query: int = 5,
                 n_train_cls:int = -1,
                 merge_task:bool = True,
                 num_worker:int = os.cpu_count(),
                 pin_memory:bool = True,
                 ) -> None:       
        
        if not isinstance(dataset, str):
            raise ValueError(f"dataset arg must be a string, \
                but found {type(dataset)} instead")
        elif dataset not in ds_map:
            raise Exception("dataset is not exist or not available")
        else:
            self.ds_name = dataset
            self.train_ds = ds_map[dataset](
                root=root,
                train=True,
                transform=transform,
                target_transform=target_transform,
                download=download,
                k_shot=k_shot,
                k_query=k_query,
                n_train_cls=n_train_cls,
                merge_task=merge_task,
                maml=True
            )
            self.test_ds = ds_map[dataset](
                root=root,
                train=True,
                transform=transform,
                target_transform=target_transform,
                download=download,
                maml=False
            )
        
        if not isinstance(num_worker, int):
            raise ValueError(f"num_worker must be an integer, \
                but found {type(num_worker)} instead")
        else:
            self.wk = num_worker
        
        
        if not isinstance(pin_memory, bool):
            raise ValueError(f"pin_memory must be a boolean, \
                but found {type(pin_memory)} instead")
        else:
            self.pm = pin_memory
        
        self.config = {
            "dataset" : self.ds_name,
            "data_root_dir" : root,
            "download" : download,
            "k_shot" : k_shot,
            "k_query" : k_query,
            "n_way" : n_train_cls,
            "num_worker" : num_worker,
            "pin_memory" : pin_memory
        }
        
        if transform is not None:
            self.config['transform'] = [x.__class__.__name__ for x in transform.transforms]
        if target_transform is not None:
            self.config['target_transofrm'] = [x.__class__.__name__ for x in target_transform.transforms]
    
    def config_export(self):
        return self.config
    
    def get_k_shot(self):
        return self.train_ds.ks
    
    def get_k_query(self):
        return self.train_ds.kq

    def get_wk(self):
        return self.wk
    
    def get_pin_mem(self):
        return self.pm