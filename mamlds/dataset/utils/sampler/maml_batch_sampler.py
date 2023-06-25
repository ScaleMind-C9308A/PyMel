import os, sys
from typing import Iterator
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-2]))
from typing import *
from dataset import MamlMnistV1

from torch.utils.data.sampler import Sampler

class MamlBatchSampler(Sampler[List[int]]):
    def __init__(self, sampler: Union[Sampler[int], Iterable[int]], 
                 data_source: MamlMnistV1 = None) -> None:
        super().__init__(data_source)
        
        self.sampler = sampler
        self.ds = data_source
        self.ks = self.ds.ks
        self.kq = self.ds.kq
        self.bs = self.ks + self.kq
    
    def __iter__(self) -> Iterator[List[int]]:
        batch = [0] * self.bs
        idx_in_batch = 0
        for idx in self.sampler:
            batch[idx_in_batch] = idx
            idx_in_batch += 1
            if idx_in_batch == self.bs:
                yield batch
                idx_in_batch = 0
                batch = [0] * self.bs
        if idx_in_batch > 0:
            yield batch[:idx_in_batch]