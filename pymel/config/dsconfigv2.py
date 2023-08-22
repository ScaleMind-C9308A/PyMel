import os, sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
from typing import *
from dataset import *
from torchvision.datasets import VisionDataset

ds_map = {
    "mnist" : MamlMnist,
    "kmnist" : MamlKMnist,
    "fmnist" : MamlFMnist
}

class DSConfigV2:
    def __init__(self,
                 train_dataset: MUL_PROC_MAML_DATASET,
                 test_dataset: VisionDataset,
                 num_worker:int = os.cpu_count(),
                 pin_memory:bool = True,
                 ) -> None:
        self.train_ds = train_dataset
        self.test_ds = test_dataset
        
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
            "dataset" : self.train_ds.__class__.__name__,
            "data_root_dir" : self.test_ds.root,
            "k_shot" : self.train_ds.ks,
            "k_query" : self.train_ds.kq,
            "n_way" : self.train_ds.nt,
            "num_worker" : num_worker,
            "pin_memory" : pin_memory
        }
        
        if self.train_ds.transform is not None:
            self.config['transform'] = [x.__class__.__name__ for x in self.train_ds.transform.transforms]
        if self.train_ds.target_transform is not None:
            self.config['target_transofrm'] = [x.__class__.__name__ for x in self.train_ds.target_transform.transforms]
        
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