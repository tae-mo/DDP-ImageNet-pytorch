import os
from glob import glob

from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from torchvision.datasets import ImageFolder

from PIL import Image

class ImageNet(Dataset):
    def __init__(self, path, transforms=None, phase="train"):
        super().__init__()
        path = os.path.join(path, phase)
        
        self.labels = {}
        dirs = os.listdir(path)
        for i, dir_name in enumerate(dirs):
            self.labels[dir_name] = i
        
        self.img_paths = glob(path+"/*/*")
        self.transforms = transforms
        self.phase = phase

        print(f"{phase.upper()} Phase Data Loaded: {len(self.img_paths)} samples")
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        path = self.img_paths[idx]
        img = Image.open(path).convert("RGB")
        label = path.split("/")[-2]
        
        if self.transforms:
            img = self.transforms(img)
        else:
            img = T.ToTensor(img)
            
        if self.phase == "train":
            return img, self.labels[label]
        else:
            return img
        
def prepare_dataset(rank, world_size, args):
    transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.ColorJitter(),
        T.RandomResizedCrop((224, 224)),
        T.ToTensor(),
        T.Normalize((0.485, 0.465, 0.406), (0.229, 0.224, 0.225))
    ])
    # train_dataset = ImageNet(args.data_path, transforms=transforms)
    train_dataset = ImageFolder(os.path.join(args.data_path, "train"), transforms)
    train_sampler = DistributedSampler(train_dataset, 
                                 num_replicas=world_size, 
                                 rank=rank, 
                                 shuffle=args.shuffle, 
                                 drop_last=args.drop_last) # DistributedSampler에 shuffle option을 주면 DataLoader에는 주면 안됨
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=args.batch_size, 
                                  pin_memory=args.pin_memory, 
                                  num_workers=args.num_workers,
                                  drop_last=args.drop_last, 
                                  sampler=train_sampler)
    
    # val_dataset = ImageNet(args.data_path, transforms=transforms, phase='val')
    val_dataset = ImageFolder(os.path.join(args.data_path, "val"), transforms)
    val_sampler = DistributedSampler(val_dataset, 
                                 num_replicas=world_size, 
                                 rank=rank, 
                                 shuffle=False, 
                                 drop_last=args.drop_last)
    val_dataloader = DataLoader(val_dataset,
                                  batch_size=args.batch_size, 
                                  pin_memory=args.pin_memory, 
                                  num_workers=args.num_workers,
                                  drop_last=args.drop_last, 
                                  sampler=val_sampler)
    return train_dataloader, val_dataloader
        
if __name__ == "__main__":
    imagenet = ImageNet("/home/data/imagenet")
    sample = next(iter(imagenet))
    
    print(f"sample:\n{sample[0].size, sample[1]}")
        
        
        
        
        

