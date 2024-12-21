# Time-step Skipping Pruning and 4-bit Quantization for Systolic Arrays
PyTorch implementation of "timestep skipping" structured pruning of convolution layers, with 4-bit quantized-aware-training applied later. 

## Weight Stationary Architecture Pruning
![ws_prune](images/ws_prune.png)
Each output channel's weight block prunes weights across select $$k_{ij}$$ indices, which are created from flattening indices $$k_i$$ and $$k_j$$ of the weight kernels of the convolution layers. For pruning sparsity ratio $$0 \leq P \lt 1$$ and length and width of the weight kernel $$l$$ and $$w$$, each output channel prunes $$\lfloor Plw \rceil$$ $$k_{ij}$$ indices out of $$lw$$ $$k_{ij}$$ indices. At each timestep $$t$$, the systolic array loads unpruned $$k_{ij}$$'s weights for each output channel. 

## Output Stationary Architecture Pruning
![os_prune](images/os_prune.png)
