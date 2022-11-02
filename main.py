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

from dataset_aihub import prepare_dataset
from train import train, valid
from utils import save_ckpt

def parse_args():
    parser = argparse.ArgumentParser(description="Imagenet Training")
    
    ## Config
    parser.add_argument("--exp", type=str, default="./exp/default")
    
    ## training
    parser.add_argument("--data-path", type=str, default="/home/data/imagenet")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--valid-iter", type=int, default=1000)
    parser.add_argument("--every", type=int, default=100)
    
    ## data loader
    parser.add_argument("--pin-memory", action='store_true')
    parser.add_argument("--num-workers", type=int, default=2) # may cause a bottleneck if set to be 0
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--imgsz", type=int, default=600)
    
    parser.add_argument("--local-rank", type=int, default=0)
    
    return parser.parse_args()

def cleanup():
    dist.destroy_process_group()
            

def main(rank, world_size, args):
    torch.cuda.set_device(rank) # set gpu id for each process

    train_loader, val_loader = prepare_dataset(rank, world_size, args)
    
    model = resnet50().to(rank)
    model = DDP(model, 
                device_ids=[rank], 
                output_device=rank)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    if rank == 0: print(f"Start Training...")
    best_acc, best_loss = 0., float("inf")
    for epoch in range(args.epochs):
        # we have to tell DistributedSampler which epoch this is
        # and guarantees a different shuffling order
        train_loader.sampler.set_epoch(epoch)
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, rank, args)
        
        val_acc, val_loss = valid(model, val_loader, criterion, rank, args)
        
        ## gather
        ## reason of using ones_like: 
        ## the container's value should be on the same device with the value it will contain
        g_acc = [torch.ones_like(val_acc) for _ in range(world_size)]
        g_loss = [torch.ones_like(val_loss) for _ in range(world_size)]
        
        dist.all_gather(g_acc, val_acc)
        dist.all_gather(g_loss, val_loss)
        
        if rank == 0:
            val_acc = torch.stack(g_acc, dim=0)
            val_loss = torch.stack(g_loss, dim=0)
            val_acc, val_loss = val_acc.mean(), val_loss.mean()
            print(f"EPOCH {epoch} VALID: acc = {val_acc}, loss = {val_loss}")
            if val_acc > best_acc:
                save_ckpt({
                    "epoch": epoch+1,
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, file_name=os.path.join(args.exp, f"best_acc.pth"))
            if val_loss < best_loss:
                save_ckpt({
                    "epoch": epoch+1,
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, file_name=os.path.join(args.exp, f"best_loss.pth"))
            save_ckpt({
                    "epoch": epoch+1,
                    "state_dict": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                }, file_name=os.path.join(args.exp, f"last.pth"))
        scheduler.step()
        dist.barrier()
        
    
    cleanup()
    
    
"""
TODO:
- consider batch size in ddp and clarify the saving policy
"""
if __name__ == "__main__":
    args = parse_args()
    args.local_rank = int(os.environ['LOCAL_RANK']) # for simplicity
    
    dist.init_process_group("nccl")

    if "./exp" not in args.exp:
        args.exp = os.path.join("./exp", args.exp)
    os.makedirs(args.exp, exist_ok=True)

    main(rank=args.local_rank, world_size=dist.get_world_size(), args=args)



