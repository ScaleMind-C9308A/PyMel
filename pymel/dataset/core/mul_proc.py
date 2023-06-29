import os, sys
from typing import *
from typing import Any, Callable
from .core import MAML_DATASET

class MUL_PROC_MAML_DATASET(MAML_DATASET):
    def __init__(self, transform: Callable[..., Any] = None, target_transform: Callable[..., Any] = None, k_shot: int = 5, k_query: int = 5, n_train_cls: int = -1, merge_task: bool = True) -> None:
        super().__init__(transform, target_transform, k_shot, k_query, n_train_cls, merge_task)
    
    def check_n_train_cls(self, nt):
        if nt != -1:            
            raise UserWarning("Currently in this version, n_train_cls cannot be used and always set to -1")
        return -1
    
    def check_merge_task(self, mt):
        if mt == False:
            raise UserWarning("Currently in this version, merge_task cannot be False and always set to True")
        return True
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self):
        return super().__getitem__()