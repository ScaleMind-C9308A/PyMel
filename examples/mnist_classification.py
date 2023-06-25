import os, sys
sys.path.append("/".join(os.path.dirname(__file__).split("/")[:-1]))
import argparse

import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms

from mamlds import MamlMnistV2
from mamlds.dataset.utils import maml_detach

def main(args: argparse):
    args.ngpus = torch.cuda.device_count()
    args.rank = 0
    args.dist_url = f'tcp://localhost:{args.port}'
    args.world_size = args.ngpus
    
    mp.spawn(main_worker, (args,), args.ngpus)

def main_worker(gpu, args):
    args.rank += gpu
    
    dist.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
    
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    train_ds = MamlMnistV2(
        root="~/data",
        train = True,
        transform=train_transform,
    )
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        prog="MINST MULTI TASK CLASSIFICATION"
    )
    
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning Rate")
    parser.add_argument("--ks", type=int, default=5,
                        help="#sample in support set")
    parser.add_argument("--kq", type=int, default=5,
                        help="#sample in query set")