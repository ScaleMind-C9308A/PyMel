import os, sys
import torch
from typing import Callable, Optional, Any, Tuple
from torchvision.datasets import MNIST
import random

class MamlMnist(MNIST):
    def __init__(self, 
                 root: str = "~/data", 
                 train: bool = True, 
                 transform: Callable[..., Any] = None, 
                 target_transform: Callable[..., Any] = None, 
                 download: bool = False,
                 k_shot: int = 5, k_query: int = 5,
                 n_train_cls:int = 5
                 ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        
        self.ks = k_shot
        self.kq = k_query
        self.nt = n_train_cls        
        
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
        return self.sample_cls_cnt // (self.ks + self.kq) - 1
    
    def __getitem__(self, index: int):
        
        if index >= len(self):
            raise ValueError("Data set index of out range")
        elif index == -1:
            index = len(self) - 1
        
        selected_cls = random.sample(list(self.dict_ds.keys()), k=self.nt)        
        
        selected_dict = {
            _cls : self.dict_ds[_cls][
                index*(self.ks + self.kq) : (index + 1)*(self.ks + self.kq)
            ] for _cls in selected_cls
        }
        
        support_dict = {
            _cls : selected_dict[_cls][:self.ks] for _cls in selected_dict
        }       
        
        query_dict = {
            _cls : selected_dict[_cls][self.ks:] for _cls in selected_dict
        }           
        
        support_x = [
            self.transform(x) if self.transform is not None else x for x in 
            [
                item for sublist in list(support_dict.values()) for item in sublist
            ]
        ]
        
        support_y = []
        for _cls in support_dict:
            support_y += [_cls]*self.ks
        
        support_y = [
            self.target_transform(x) if self.target_transform is not None else x for x in support_y
        ]
        
        query_x = [
            self.transform(x) if self.transform is not None else x for x in 
            [
                item for sublist in list(query_dict.values()) for item in sublist  
            ]
        ]
        
        query_y = []
        for _cls in query_dict:
            query_y += [_cls]*self.ks
        
        query_y = [
            self.target_transform(x) if self.target_transform is not None else x for x in query_y
        ]
        
        return support_x, torch.LongTensor(support_y), query_x, torch.LongTensor(query_y)

if __name__ == "__main__":
    ds = MamlMnist(train=False, download=True)
    
    print(f"len dataset: {len(ds)}, sample per class: {ds.sample_cls_cnt}")    
    
    for x in range(len(ds)):
        sample_s_x, sample_s_y, sample_q_x, sample_q_y = ds[x]
        
        print(len(sample_s_x), len(sample_s_y), len(sample_q_x), len(sample_q_y))        