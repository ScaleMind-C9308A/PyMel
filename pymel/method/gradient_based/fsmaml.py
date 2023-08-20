import os, sys
from pymel.config import DSConfig, TrainConfig
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
from core import Trainer, opt_mapping
from dataset.utils import maml_detach
import torch
from torch import nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from random import randint
import argparse
from tqdm import tqdm
import copy


class FSMAML(Trainer):
    def __init__(self, ds_cfg: DSConfig, tr_cfg: TrainConfig, 
                 model: nn.Module = None, gpus: list[int] = ...,
                 meta_opt: str = None, 
                 meta_lr: float = 0.001,
                 meta_wd: float = 1e-4,
                 sp_opt: str = None,
                 sp_lr: float = 0.01,
                 sp_wd: float = 1e-4,
                 meta_criterion: torch.nn.Module = nn.BCELoss(),
                 clf_criterion: torch.nn.Module = nn.CrossEntropyLoss(),
                 epoch: int = 100
                 ) -> None:
        super().__init__(ds_cfg, tr_cfg, model, gpus)
        
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
        
        self.ds_cfg = ds_cfg
        self.tr_cfg = tr_cfg
        self.model = model
        self.gpus = gpus
        
        for optn, opt in zip(["meta_opt", "sp_opt"], [meta_opt, sp_opt]):
            if not isinstance(opt, str):
                raise TypeError(f"PyMel GPT: {optn} must be a string, \
                    but found {type(opt)} instead")
            elif opt is None:
                raise ValueError(f"PyMel GPT: {optn} cannot be a None")
        
        for name, value in zip(
            ["meta_lr", "meta_wd", "sp_lr", "sp_wd"],
            [meta_lr, meta_wd, sp_lr, sp_wd]
        ):
            if not isinstance(value, float):
                raise TypeError(f"PyMel GPT: {name} must be a float, \
                    but found {type(value)} instead")
        
        for critn, crit in zip(
            ["meta_criterion", "clf_criterion"], [meta_criterion, clf_criterion]
        ):
            if not isinstance(crit, torch.nn.Module):
                raise TypeError(f"PyMel GPT: {critn} must be a float, \
                    but found {type(crit)} instead")
        
        if not isinstance(epoch, int):
            raise TypeError(f"PyMel GPT: {epoch} must be an int, \
                    but found {type(epoch)} instead")
        
        self.meta_opt = meta_opt
        self.sp_opt = sp_opt
        self.meta_lr = meta_lr
        self.meta_wd = meta_wd
        self.sp_lr = sp_lr
        self.sp_wd = sp_wd
        self.meta_crit = meta_criterion
        self.clf_crit = clf_criterion
        self.epoch
        self.method = "fsmaml"
        self.tr_cfg.folder_setup(method=self.method)
        
    def train(self, port=randint(1000, 8000)):
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        
        args.ngpus = torch.cuda.device_count()
        args.rank = 0
        args.dist_url = f'tcp://localhost:{args.port}'
        print(f"PyMel GPT: The experiment is deployed at {args.dist_url}")
        args.world_size = args.ngpus
        
        mp.spawn(self.main_worker, (args,), args.ngpus)
        
    def main_worker(self, gpu, args):
        args.rank += gpu
    
        dist.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
        
        torch.cuda.set_device(gpu)
        torch.backends.cudnn.benchmark = True
        
        batch_size = self.ds_cfg.get_k_shot() + self.ds_cfg.get_k_query()
        assert batch_size % args.world_size == 0
        
        train_sampler = DistributedSampler(self.ds_cfg.train_ds)
        
        per_device_batch_size = batch_size // args.world_size
        per_device_k_shot = self.ds_cfg.get_k_shot() // args.world_size
        per_device_k_query = self.ds_cfg.get_k_query() // args.world_size
        
        train_dl = DataLoader(
            dataset=self.ds_cfg.train_ds, 
            batch_size=per_device_batch_size, 
            num_workers=self.ds_cfg.get_wk(), 
            pin_memory=self.ds_cfg.get_pin_mem(), 
            sampler=train_sampler
        )
        
        test_dl = DataLoader(
            dataset=self.ds_cfg.test_ds,
            batch_size=1,
            num_workers=self.ds_cfg.get_wk(), 
            pin_memory=self.ds_cfg.get_pin_mem()
        )
        
        global_model = self.model.cuda(gpu)
        global_model = nn.SyncBatchNorm.convert_sync_batchnorm(global_model)
        global_model = torch.compile(model=global_model)
        global_model = DDP(global_model, device_ids=[gpu])
        
        meta_optimizer = opt_mapping[self.meta_opt](
            global_model.parameters(), 
            lr=self.meta_lr, weight_decay=self.meta_wd
        )
        
        for epoch in range(self.epoch):
            global_model.train()
            
            for train_idx, data_dict in enumerate(train_dl):
            
                metaloss = 0.0                
                
                for task in data_dict:
                    model=copy.deepcopy(global_model)
                    model.train()
                    
                    sp_optimizer = opt_mapping[self.sp_opt](
                        model.parameters(), 
                        lr=self.sp_lr, weight_decay=self.sp_wd
                    )
                    
                    sp_x, sp_y, qr_x, qr_y = maml_detach(
                        batch_dict=data_dict,
                        k_shot=per_device_k_shot,
                        k_query=per_device_k_query,
                        task=task
                    )
                    
                    sp_x = sp_x.cuda(gpu, non_blocking=True)
                    sp_y = sp_y.cuda(gpu, non_blocking=True)
                    sp_logits = model(sp_x)
                    sp_loss = self.meta_crit(sp_logits[:, task], sp_y)
                    sp_optimizer.zero_grad()
                    sp_loss.backward()
                    sp_optimizer.step()
                    
                    qr_x = qr_x.cuda(gpu, non_blocking=True)
                    qr_y = qr_y.cuda(gpu, non_blocking=True)
                    qr_logits = model(qr_x)
                    qr_loss = self.meta_crit(qr_logits[:, task], qr_y)
                    metaloss += qr_loss
                
                meta_optimizer.zero_grad()
                metagrads=torch.autograd.grad(
                    metaloss, list(global_model.parameters()), allow_unused=True
                )
                for w,g in zip(list(global_model.parameters()), metagrads):
                    w.grad=g 
                meta_optimizer.step()
            
            if args.rank == 0:
                global_model.eval()
                with torch.no_grad():
                    test_loss = 0
                    correct = 0
                    total = 0
                    batch_count = 0
                    for test_idx, (test_imgs, test_labels) in enumerate(test_dl):
                        batch_count = test_idx
                        test_imgs = test_imgs.cuda(gpu, non_blocking=True)
                        test_labels = test_labels.cuda(gpu, non_blocking=True)
                        test_logits = model(test_imgs)                
                    
                        test_loss += self.clf_crit(test_logits, test_labels).item()
                        _, predicted = test_logits.max(1)
                        total += test_labels.size(0)
                        correct += predicted.eq(test_labels).sum().item()
                        
                print(f"Epoch: {epoch} - MetaLoss: {metaloss.item()/self.ds_cfg.train_ds.nt} - \
                    Test Loss: {test_loss/batch_count} - Test Acc: {100*correct/total}%")  