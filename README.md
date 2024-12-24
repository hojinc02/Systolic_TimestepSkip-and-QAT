# Time-step Skipping Pruning and 4-bit Quantization for Systolic Arrays
PyTorch implementation of "timestep skipping" structured pruning of convolution layers, with 4-bit quantized-aware-training applied later. 

## Weight Stationary Architecture Pruning
![ws_prune](images/ws_prune.png)
For each output channel in the convolution layer, prune the weights at specific $$k_{ij}$$​ indices. These $$k_{ij}$$​​ indices represent the flattened positions within the weight kernel, derived from the spatial positions $$k_i$$​ and $$k_j$$​. For a pruning sparsity ratio $$P$$, and a weight kernel of dimensions $$l$$ (length) and $$w$$ (width), $$\lfloor P \cdot l \cdot w \rceil$$ indices are pruned per output channel. At each timestep, the unpruned $$k_{ij}$$​ index weights are loaded onto the systolic array for processing for each output channel.

## Output Stationary Architecture Pruning
![os_prune](images/os_prune.png)

## Results
| Type           | WS Test Accuracy | OS Test Accuracy |
| -------------- | ---------------- | ---------------- |
| Original       | 90.98%                              |
| Pruned         | 90.77%           | 89.32%           |
| Pruned + QAT   | 90.79%           | 88.28%           |
| Error Increase |  0.19%           |  2.70%           |
