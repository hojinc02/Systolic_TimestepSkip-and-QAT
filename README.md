# Time-step Skipping Pruning and 4-bit Quantization for Systolic Arrays
A PyTorch implementation of structured pruning for convolutional layers using "timestep skipping," followed by 4-bit quantization-aware training optimized for 2D systolic arrays.

## Weight Stationary Architecture Pruning
![ws_prune](images/ws_prune.png)
For each output channel in a convolution layer, prune weights at specific $$k_{ij}$$​ indices using the $$L_n$$-norm, ensuring the same number of $$k_{ij}$$ are pruned per output channel. These $$k_{ij}$$​​ indices correspond to the flattened positions within the weight kernel, derived from spatial positions $$k_i$$​ and $$k_j$$​. For a pruning sparsity ratio $$P$$, and a weight kernel of dimensions $$l$$ (length) and $$w$$ (width), $$\lfloor P \cdot l \cdot w \rceil$$ indices are pruned per output channel. At each timestep, the unpruned $$k_{ij}$$​ index weights are loaded onto the systolic array for processing for each output channel. The partial sum (psum) can either be accumulated during array processing and summed later or summed in parallel with array processing.

## Output Stationary Architecture Pruning
![os_prune](images/os_prune.png)
In a convolution layer, $$L_n$$-norm across output channels is used to prune $$\lfloor P \cdot l \cdot w \cdot I \cdot O\rceil$$ weights per $$k_{ij}$$ index for each input channel, where $$I$$ and $$O$$ are the input and output channel counts. Only unpruned weights are processed in paralleled with the inputs. Input formatting ensures the sliding window for the convolution kernel includes only the unpruned weight positions per input channel. $$n_{ij}'$$ denotes the output pixel index. 

## Results
The results were obtained using a small **14M** parameter VGG16 model, which achieved 90.98% accuracy on CIFAR-10 when trained from scratch with the AdamW optimizer. The table below summarizes the test accuracy after pruning **80%** of the convolutional layer weights, applying 4-bit quantization-aware training (QAT), and the resulting error delta compared to the original full-precision model. Pruning was performed iteratively during training using a scheduler to optimize the weight reduction process.
| Type           | WS Test Accuracy | OS Test Accuracy |
| -------------- | ---------------: | ---------------: |
| Full           | *90.98%*         | *90.98%*         |
| Pruned         | 90.77%           | 89.32%           |
| Pruned + QAT   | 90.79%           | 88.28%           |
| Error Delta    | **-0.19%**       | **-2.70%**       |

## Benefits
Pruning reduces the processing requirements and model size by approximately a factor of $$1/(1−S)$$, where $$0 \leq S \lt 1$$ represents sparsity. OS pruning enables finer-grained sparsity by allowing each input channel to prune any number of $$k_{ij}$$ positions, offering greater flexibility compared to WS pruning.

## Further Research
Fine-grained WS pruning could be achieved by allowing a variable number of output channels' psum blocks to be computed at each timestep. However, this approach is incompatible with the parallel accumulation performed by the special function units (SFU) in the current simple systolic array design. Future research should focus on addressing the challenge of fast accumulation to enable finer-grained WS pruning.
