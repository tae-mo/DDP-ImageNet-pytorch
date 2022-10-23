# Training ImageNet using DDP pytorch

ref: https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51
### usage
```
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python main.py --pin-memory --shuffle --exp <your-exp-name>
```
