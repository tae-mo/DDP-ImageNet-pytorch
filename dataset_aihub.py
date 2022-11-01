import os
from glob import glob

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class TrashDataset(Dataset):
    def __init__(self, path, transforms=None, phase="train"):
        super().__init__()
        path = os.path.join(path, phase) # path/phase
        
        self.labels = {}
        dirs = os.listdir(path)
        for i, dir_name in enumerate(dirs):
            self.labels[dir_name] = i
        
        self.img_paths = sorted(glob(path+"/images/*"))
        self.label_paths = sorted(glob(path+"/labels/*"))
        self.transforms = transforms
        self.phase = phase
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = self.label_paths[idx]
        
        img = Image.open(img_path).convert("RGB")
        with open(label_path, "r") as f:
            label = int(f.read().strip().split(" ")[0])
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor()(img)
            
        return img, label
        
def prepare_dataset(rank, world_size, args):
    assert not args.batch_size % world_size, "batch size must be dividable by world size"
    batch_per_gpu = args.batch_size // world_size
    if rank == 0:
        print(f"{batch_per_gpu} batches per GPU...")
    
    transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(),
        T.RandomResizedCrop((args.imgsz, args.imgsz)),
        T.ToTensor(),
        T.Normalize((0.485, 0.465, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = TrashDataset(args.data_path, transforms=transforms)
    val_dataset = TrashDataset(args.data_path, transforms=transforms, phase='test')
    
    # train_dataset = ImageFolder(os.path.join(args.data_path, "Training", "images"), transforms)
    # val_dataset = ImageFolder(os.path.join(args.data_path, "Validation", "images"), transforms)
    
    train_sampler = DistributedSampler(train_dataset) # DistributedSampler에 shuffle option을 주면 DataLoader에는 주면 안됨
    val_sampler = DistributedSampler(val_dataset, shuffle=False, drop_last=args.drop_last)
    
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_per_gpu, 
                                  pin_memory=args.pin_memory, 
                                  num_workers=args.num_workers,
                                  drop_last=args.drop_last, 
                                  sampler=train_sampler)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=batch_per_gpu, 
                                  pin_memory=args.pin_memory, 
                                  num_workers=args.num_workers,
                                  drop_last=args.drop_last, 
                                  sampler=val_sampler)
    return train_dataloader, val_dataloader
        
if __name__ == "__main__":
    trash_dataset = TrashDataset("/media/data1/taejune/TrashDataset")
    sample = next(iter(trash_dataset))
    
    print(f"sample:\n{sample[0].size(), sample[1]}")
