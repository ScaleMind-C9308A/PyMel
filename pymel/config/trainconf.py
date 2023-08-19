import os, sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
from typing import *
from torch import nn

class TrainConfig:
    def __init__(self,
                 checkpoint: bool = True,
                 logging: bool = True,
                 save_dir: str = None,
                 save_best: bool = True,
                 extension:str = "pt",
                 save_last: bool = False
                 ) -> None:
        
        for _varn, _var in zip(["checkpoint", "logging"], [checkpoint, logging]):
            if not isinstance(_var, bool):
                raise TypeError(f"{_varn} must be a boolean, \
                    but found {type(_var)} instead")
        self.cp = checkpoint
        self.lg = logging
        
        if self.cp or self.lg:
            if save_dir is None:
                self.sv_dir = os.getcwd() + "/pymel_benchmark"
                if not os.path.exists(self.sv_dir):
                    os.mkdir(self.sv_dir)
            else:
                if not isinstance(save_dir, str):
                    raise TypeError(f"save_dir must be a string \
                        but found {type(save_dir)} instead")
                elif not os.path.exists(save_dir):
                    raise FileExistsError(f"the provided save_dir path: \
                        f{save_dir} is not exist")
                else:
                    self.sv_dir = save_dir + "/pymel_benchmark"
                    if not os.path.exists(self.sv_dir):
                        os.mkdir(self.sv_dir)                    
        
        if self.cp:
            self.sv_best = save_best
            self.sv_last = save_last
            self.ext = extension
            
    @staticmethod
    def folder_setup(self, method):
        self.method_dir = self.sv_dir + f"/{method}"
        if not os.path.exists(self.method_dir):
            os.mkdir(self.method_dir)
        sub_dirs = os.listdir(self.method_dir)
        
        self.exp_dir = self.method_dir + f"/ext{len(sub_dirs)}"
        if not os.path.exists(self.exp_dir):
            os.mkdir(self.exp_dir)
    
    @staticmethod
    def checkpoint(self):
        return self.cp
    
    @staticmethod
    def get_sv_best(self):
        return self.sv_best
    
    @staticmethod
    def get_sv_last(self):
        return self.sv_last
    
    @staticmethod
    def get_ext(self):
        return self.ext
    
    @staticmethod
    def get_sv_dir(self):
        return self.sv_dir

    @staticmethod
    def config_export(self):
        return {
            "checkpoint" : self.cp,
            "logging" : self.lg,
            "save_dir" : self.sv_dir,
            "save_best" : self.sv_best,
            "save_last" : self.sv_last,
            "extension" : self.ext
        }