import torch
from torch import nn
from torch.nn.utils import prune

from quant_layer import QuantConv2d

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
    num_sticks = num_oc * k1 * k2
    num_prune_sticks = round(sparsity * num_sticks)

    print(f'Pruning {num_prune_sticks} kij-sticks out of {num_sticks} kij-sticks ({num_prune_sticks/num_sticks:.1%} pruned)')

    mask = torch.empty(conv_layer.weight.shape, dtype=torch.bool, device=conv_layer.weight.device)
    with torch.no_grad(): 
        num_already_pruned = (conv_layer.weight_mask[:,0,:,:]==0).sum().item() if hasattr(conv_layer, 'weight_mask') else 0
        norms = torch.norm(conv_layer.weight, p=ln, dim=(1), keepdim=True)
        threshold = norms.view((-1)).topk(k=num_prune_sticks+num_already_pruned, largest=False).values[-1]
        mask= (norms > threshold).expand(num_oc, num_ic, k1, k2)
        prune.custom_from_mask(conv_layer, 'weight', mask)

def os_conv_prune(conv_layer:nn.Conv2d, sparsity:float, ln:int=1): 
    num_oc, num_ic, k1, k2 = conv_layer.weight.shape
    num_sticks = num_ic * k1 * k2
    num_prune_sticks = round(sparsity * num_sticks)

    print(f'Pruning {num_prune_sticks} kij-sticks out of {num_sticks} kij-sticks ({num_prune_sticks/num_sticks:.1%} pruned)')

    mask = torch.empty(conv_layer.weight.shape, dtype=torch.bool, device=conv_layer.weight.device)
    with torch.no_grad(): 
        num_already_pruned = (conv_layer.weight_mask[0,:,:,:]==0).sum().item() if hasattr(conv_layer, 'weight_mask') else 0
        norms = torch.norm(conv_layer.weight, p=ln, dim=(0), keepdim=True)
        threshold = norms.view((-1)).topk(k=num_prune_sticks+num_already_pruned, largest=False).values[-1]
        mask= (norms > threshold).expand(num_oc, num_ic, k1, k2)
        prune.custom_from_mask(conv_layer, 'weight', mask)

def quantize_pruned(model): 
    for i, l in enumerate(model.features): 
        if isinstance(l, nn.Conv2d) and hasattr(l, 'weight_mask'): 
            ql = QuantConv2d(l.in_channels, l.out_channels, l.kernel_size[0], l.stride, 
                            l.padding, l.dilation, l.groups, l.bias, bits=4).cuda()
            ql.weight.data = l.weight.data.detach().clone() # copy original weights
            prune.custom_from_mask(ql, 'weight', l.weight_mask) # Add pruning parameters here with the correct weight_mask
            model.features[i] = ql # replace layer

if __name__=='__main__': 
    from quant_layer import QuantConv2d

    dd = nn.Conv2d(2,2,3)
    print(dd.weight)
    os_conv_prune(dd,1/9,1)
    print(dd.weight)
    os_conv_prune(dd,1/9,1)
    print(dd.weight)
    os_conv_prune(dd,1/9,1)
    print(dd.weight)
    os_conv_prune(dd,1/9,1)
    print(dd.weight)
