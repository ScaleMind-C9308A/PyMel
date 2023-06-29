import os, sys
from typing import *

class MAML_DATASET:
    def __init__(self) -> None:
        pass
    
    def check_target_transform(self, target_transform):
        if target_transform is not None:
            raise UserWarning("Target Transform is not used in MAML DATASET") 
    
    def check_int_arg(self, *args, **kwargs):
        for arg in kwargs:
            if not isinstance(kwargs[arg], int):
                raise ValueError(f"{arg} must be integer type but found {type(kwargs[arg])} instead")
    def check_bool_arg(self, *args, **kwargs):
        for arg in kwargs:
            if not isinstance(kwargs[arg], bool):
                raise ValueError(f"{arg} must be boolean type but found {type(kwargs[arg])} instead")
    
    def __len__(self):
        raise NotImplementedError
    
    def __getitem__(self):
        raise NotImplementedError