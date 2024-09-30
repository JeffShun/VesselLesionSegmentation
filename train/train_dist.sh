
#!/bin/bash
torchrun --nproc_per_node=4 train_multi_gpu.py --task IA
# torchrun --nproc_per_node=4 train_multi_gpu.py --task IA
