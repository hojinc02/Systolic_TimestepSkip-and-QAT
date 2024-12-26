import torch
from torch import nn
from torch.backends import cudnn
from torch.nn.utils import prune

import torchvision
import torchvision.transforms as transforms

import time
import os
import shutil

from pyfiles.quant_layer import QuantConv2d

print_freq = 100
batch_size = 64

normalize = transforms.Normalize(mean=[0.491, 0.482, 0.447], std=[0.247, 0.243, 0.262])
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ]))
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def attach_prune(model): 
    # Attaches pruning parameters to layers
    os_prune_vgg16(model, 0.5)

def ws_prune_vgg16(model, prune_percentage:float): 
    first_conv = True
    for l in model.features: 
        if isinstance(l, nn.Conv2d) and not first_conv: 
            ws_conv_prune(l, prune_percentage, ln=1)
        elif isinstance(l, nn.Conv2d) and first_conv: 
            first_conv = False # Skip first conv layer

def os_prune_vgg16(model, prune_percentage:float): 
    first_conv = True
    for l in model.features: 
        if isinstance(l, nn.Conv2d) and not first_conv: 
            os_conv_prune(l, prune_percentage, ln=1)
        elif isinstance(l, nn.Conv2d) and first_conv: 
            first_conv = False # Skip first conv layer

def print_sparsity(model): 
    for i, l in enumerate(model.features): 
        if isinstance(l, nn.Conv2d) and hasattr(l, 'weight_mask'): 
            print(f'layer {i} sparsity: {(l.weight_mask==0).sum()/l.weight_mask.numel():.3f}')

def ws_conv_prune(conv_layer:nn.Conv2d, sparsity:float, ln:int=1): 
    num_oc, num_ic, k1, k2 = conv_layer.weight.shape
    num_sticks = k1 * k2
    num_prune_sticks = round(sparsity * num_sticks)

    print(f'WS Pruning Layer: {num_prune_sticks/num_sticks:.1%} pruned')

    mask = torch.empty(conv_layer.weight.shape, dtype=torch.bool, device=conv_layer.weight.device)
    with torch.no_grad(): 
        for oc in range(num_oc): 
            num_already_pruned = (conv_layer.weight_mask[oc,0,:,:]==0).sum().item() if hasattr(conv_layer, 'weight_mask') else 0
            norms = torch.norm(conv_layer.weight[oc,:,:,:], p=ln, dim=(0), keepdim=True)
            threshold = norms.view((-1)).topk(k=num_prune_sticks+num_already_pruned, largest=False).values[-1]
            mask[oc,:,:,:] = (norms > threshold).expand(num_ic, k1, k2)
        prune.custom_from_mask(conv_layer, 'weight', mask)

def os_conv_prune(conv_layer:nn.Conv2d, sparsity:float, ln:int=1): 
    num_oc, num_ic, k1, k2 = conv_layer.weight.shape
    num_sticks = num_ic * k1 * k2
    num_prune_sticks = round(sparsity * num_sticks)

    print(f'OS Pruning Layer: {num_prune_sticks/num_sticks:.1%} pruned')

    mask = torch.empty(conv_layer.weight.shape, dtype=torch.bool, device=conv_layer.weight.device)
    with torch.no_grad(): 
        num_already_pruned = (conv_layer.weight_mask[0,:,:,:]==0).sum().item() if hasattr(conv_layer, 'weight_mask') else 0
        norms = torch.norm(conv_layer.weight[:,:,:,:], p=ln, dim=(0), keepdim=True)
        threshold = norms.view((-1)).topk(k=num_prune_sticks+num_already_pruned, largest=False).values[-1]
        mask[:,:,:,:] = (norms > threshold).expand(num_oc, num_ic, k1, k2)
        prune.custom_from_mask(conv_layer, 'weight', mask)

def quantize_pruned(model): 
    for i, l in enumerate(model.features): 
        if isinstance(l, nn.Conv2d) and hasattr(l, 'weight_mask'): 
            ql = QuantConv2d(l.in_channels, l.out_channels, l.kernel_size[0], l.stride, 
                             l.padding, l.dilation, l.groups, l.bias, bits=4).cuda()
            ql.weight.data = l.weight.data.detach().clone().cuda() # copy original weights
            prune.custom_from_mask(ql, 'weight', l.weight_mask) # Add pruning parameters here with the correct weight_mask
            model.features[i] = ql # replace layer

def train_epoch(model, criterion, optimizer, epoch):

    model.train()
    
    loss_accum = 0
    num_correct = 0
    num_processed = 0
    end = time.time()
    for i, (input, target) in enumerate(trainloader, start=1):
        
        input, target = input.cuda(), target.cuda()

        output:torch.Tensor = model(input)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_accum += loss.item()
        num_correct += output.argmax(dim=1).eq(target).sum().item()
        num_processed += output.size(dim=0)
        
        if i % print_freq == 0 or i == len(trainloader):
            print(f'Epoch: {epoch} ({i}/{len(trainloader)})\t'
                  f'Time Elapsed: {(time.time() - end):.1f}s\t'
                  f'Loss: {loss_accum/num_processed:.2e}\t'
                  f'Accuracy: {num_correct/num_processed:.2%}')
            loss_accum = 0
            num_correct = 0
            num_processed = 0

def save_checkpoint(state, is_best, fdir):
    filepath = os.path.join(fdir, 'checkpoint.pth')
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(fdir, 'model_best.pth.tar'))

def train_model(model, fdir, criterion, optimizer, epochs, prune_schedule:dict=None, start=0, stype='ws'): 
    
    os.makedirs(fdir, exist_ok=True)

    best_prec = 0

    model.cuda()
    criterion = criterion.cuda()
    cudnn.benchmark = True

    for epoch in range(start, epochs):

        if prune_schedule is not None and epoch in prune_schedule: 
            if stype=='ws': 
                ws_prune_vgg16(model, prune_schedule[epoch])
            elif stype=='os': 
                os_prune_vgg16(model, prune_schedule[epoch])
            else: 
                raise Exception('stype unrecognized')

        train_epoch(model, criterion, optimizer, epoch)
        prec = val_model(model)
        
        if prune_schedule is not None and epoch in prune_schedule: 
            is_best = True
            best_prec = prec
        else: 
            is_best = prec > best_prec
            best_prec = max(prec, best_prec)
        print(f'Best Test Accuracy Since Last Pruning: {best_prec:.2%}')
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_prec': best_prec,
            'optimizer': optimizer.state_dict(),
            'prune_schedule': prune_schedule
        }, is_best, fdir)

def val_model(model): 
    model.cuda()
    model.eval()

    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.cuda(), target.cuda()
            output = model(data)
            correct += output.argmax(dim=1).eq(target).sum().item()
    
    acc = correct/len(testloader.dataset)

    print(f'Test Set Accuracy: {acc:.2%}')

    return acc

def count_param(model): 
    count = 0
    for l in model.features: 
        if hasattr(l, 'weight_mask'): 
            count += int(l.weight_mask.sum().item())
        elif hasattr(l, 'weight'): 
            count += l.weight.numel()
    return count
