# Time-step Skipping Pruning and 4-bit Quantization for Systolic Arrays
PyTorch implementation of "timestep skipping" structured pruning of convolution layers, with 4-bit quantized-aware-training applied later. 

## Weight Stationary Architecture Pruning
![ws_prune](images/ws_prune.png)
Each output channel's weight block prunes weights across select $$k_{ij}$$ indices, which are created from flattening indices ki and kj of the weight kernels of the convolution layers. Each output channel prunes The systolic array loads unpruned weigh

## Output Stationary Architecture Pruning
![os_prune](images/os_prune.png)
