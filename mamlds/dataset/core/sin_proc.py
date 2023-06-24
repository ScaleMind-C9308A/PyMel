import os, sys
from typing import *
from typing import Any, Callable
from .core import MAML_DATASET

class SIN_PROC_MAML_DATASET(MAML_DATASET):
    def __init__(self, transform: Callable[..., Any] = None, target_transform: Callable[..., Any] = None, k_shot: int = 5, k_query: int = 5, n_train_cls: int = -1, merge_task: bool = True) -> None:
        super().__init__(transform, target_transform, k_shot, k_query, n_train_cls, merge_task)
        
    def __len__(self):
        return super().__len__()
    
    def __getitem__(self):
        return super().__getitem__()