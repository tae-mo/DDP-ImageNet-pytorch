import torch
import time

def train(model, train_loader, criterion, optimizer, rank, args):
    model.train()
    running_loss, logging_loss, train_acc = 0, 0, 0
    for step, (img, label) in enumerate(train_loader, 1):
        if args.pin_memory:
            img, label = img.to(rank, non_blocking=True), label.to(rank, non_blocking=True)    
        else:
            img, label = img.to(rank), label.to(rank)
        
        out = model(img)
        loss = criterion(out, label)
        
        running_loss += loss.item()
        logging_loss += loss.item()
        if rank == 0:
            if not step % args.every: 
                print(f"[{step}/{len(train_loader)}] loss: {logging_loss / args.every}")
                logging_loss = 0
        
        train_acc += (out.detach().argmax(-1) == label).float().sum() / args.batch_size
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return running_loss / len(train_loader), train_acc / len(train_loader)
        

def valid(model, val_loader, criterion, rank, args):
    model.eval()
    val_loss, val_acc = 0, 0
    with torch.no_grad():
        for img, label in val_loader:
            img, label = img.to(rank), label.to(rank)

            out = model(img)
            loss = criterion(out, label)
            
            val_acc += (out.detach().argmax(-1) == label).float().sum() / args.batch_size
            val_loss += loss#.item()
            
        return val_acc / len(val_loader), val_loss / len(val_loader)
        