import torch
import time

def train(model, train_loader, criterion, optimizer, rank, args):
    model.train()
    for step, (img, label) in enumerate(train_loader):
        if args.pin_memory:
            img, label = img.to(rank, non_blocking=True), label.to(rank, non_blocking=True)    
        else:
            img, label = img.to(rank), label.to(rank)
        
        out = model(img)
        loss = criterion(out, label)
        
        acc = (out.detach().argmax(-1) == label).float().sum() / args.batch_size
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

def valid(model, val_loader, criterion, rank, args):
    model.eval()
    total_loss, total_acc = 0, 0
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(rank), label.to(rank)

            out = model(img)
            loss = criterion(out, label)
            
            total_acc += (out.detach().argmax(-1) == label).float().sum() / args.batch_size
            total_loss += loss.item()
            
        return total_acc / len(val_loader), total_loss / len(val_loader)
        