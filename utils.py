import torch

def save_ckpt(state, file_name="./exp/default/default.pth"):
    torch.save(state, file_name)