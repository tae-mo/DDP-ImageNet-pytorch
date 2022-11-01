# Training ImageNet using DDP pytorch

ref: https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
### usage
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --nproc_per_node 8 main.py --pin-memory --shuffle --exp aihub_trash_classification --batch-size 64 --data-path /media/data1/taejune/TrashDataset/ --imgsz 600 --every 1000
```
