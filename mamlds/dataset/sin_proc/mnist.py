import os, sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
import torch
from typing import Callable, Optional, Any, Tuple
from torchvision.datasets import MNIST
import random
from torch.utils.data import DataLoader
from core import SIN_PROC_MAML_DATASET

class MamlMnistV1(MNIST, SIN_PROC_MAML_DATASET):
    def __init__(self, 
                 root: str = "~/data", 
                 train: bool = True, 
                 transform: Callable[..., Any] = None, 
                 target_transform: Callable[..., Any] = None, 
                 download: bool = False,
                 k_shot: int = 5, k_query: int = 5,
                 n_train_cls:int = -1,
                 merge_task:bool = True
                 ) -> None:
        super().__init__(root, train, transform, target_transform, download)
        
        # Check
        self.check_int_arg(k_shot=k_shot, k_query=k_query, n_train_cls=n_train_cls)
        self.check_bool_arg(merge_task=merge_task)
        
        self.ks = k_shot
        self.kq = k_query
        self.nt = len(self.class_to_idx) if n_train_cls == -1 else n_train_cls
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
        return self.sample_cls_cnt // (self.ks + self.kq) - 1
    
    def __getitem__(self, index: int):
        
        if index >= len(self):
            raise ValueError("Data set index of out range")
        elif index == -1:
            index = len(self) - 1
        
        selected_cls = random.sample(
            list(self.dict_ds.keys()), 
            k = len(self.class_to_idx) if self.nt == -1 else self.nt
        )        
        
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
        
        if not self.mt:
            
            support_dict = {
                _cls : self.transform(x) if self.transform is not None else x for x in support_dict[_cls]
            }
            
            query_dict = {
                _cls : self.transform(x) if self.transform is not None else x for x in query_dict[_cls]
            }
            
            return (support_dict, query_dict)
        else:         
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
            
            return torch.stack(support_x), torch.LongTensor(support_y), torch.stack(query_x), torch.LongTensor(query_y)

# """
if __name__ == "__main__":
    ds = MamlMnistV1(train=False, download=True, n_train_cls=5)
    
    print(f"len dataset: {len(ds)}, sample per class: {ds.sample_cls_cnt}")    
    
    for x in range(len(ds)):
        sample_s_x, sample_s_y, sample_q_x, sample_q_y = ds[x]
        
        print(len(sample_s_x), len(sample_s_y), len(sample_q_x), len(sample_q_y))     
    
    # Check compatibility with DataLoader
    dl = DataLoader(
        dataset=ds,
        batch_size=8,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )  
    
    print(f"Len Data Loader: {len(dl)}")
    
    for idx, (sample_s_x, sample_s_y, sample_q_x, sample_q_y) in enumerate(dl):
        print(sample_s_x.shape, sample_s_y.shape, sample_q_x.shape, sample_q_y.shape)

# """