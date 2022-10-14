import os
import argparse
from sched import scheduler

import numpy as np

import torch
import torch.nn as nn

import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from torchvision.models import resnet50
from torchvision import transforms as T

from dataset import prepare_dataset
from train import train, valid
from utils import save_ckpt

def setup(rank, world_size, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group("nccl", init_method=args.dist_url,rank=rank, world_size=world_size)
    
def parse_args():
    parser = argparse.ArgumentParser(description="Imagenet Training")
    
    ## Config
    parser.add_argument("--exp", type=str, default="./exp/default")
    
    ## DDP
    parser.add_argument("--dist_url", type=str, default="env://")
    
    ## training
    parser.add_argument("--data_path", type=str, default="/home/data/imagenet")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--valid-iter", type=int, default=1000)
    
    ## data loader
    parser.add_argument("--pin-memory", action='store_true')
    parser.add_argument("--num-workers", type=int, default=2) # may cause a bottleneck if set to be 0
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    
    return parser.parse_args()

def cleanup():
    dist.destroy_process_group()
            

def main(rank, world_size, args):
    setup(rank, world_size, args)
    torch.cuda.set_device(rank) # set gpu id for each process

    train_loader, val_loader = prepare_dataset(rank, world_size, args)
    
    model = resnet50().to(rank)
    model = DDP(model, 
                device_ids=[rank], 
                output_device=rank)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    best_acc, best_loss = 0., float("inf")
    for epoch in range(args.epochs):
        # we have to tell DistributedSampler which epoch this is
        # and guarantees a different shuffling order
        train_loader.sampler.set_epoch(epoch)
        train(model, train_loader, criterion, optimizer, rank, args)
        if rank == 0:
            epoch_acc, epoch_loss = valid(model, val_loader, criterion, rank, args)
            print(f"EPOCH {epoch}: acc = {epoch_acc}, loss = {epoch_loss}")
            if epoch_acc > best_acc:
                save_ckpt({
                    "epoch": epoch+1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, file_name=os.path.join(args.exp, f"resnet50_bestACC_epoch{epoch+1}"))
            if epoch_loss < best_loss:
                save_ckpt({
                    "epoch": epoch+1,
                    "state_dict": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, file_name=os.path.join(args.exp, f"resnet50_bestLOSS_epoch{epoch+1}"))
        scheduler.step()
    cleanup()
    
    
"""
TODO:
- add world_size & local_rank in argument
- consider batch size in ddp and clarify the saving policy
- 
"""
if __name__ == "__main__":
    world_size = 8
    args = parse_args()

    os.makedirs(args.exp, exist_ok=True)
    
    mp.spawn(
        main,
        args=(world_size, args),
        nprocs=world_size
    )



