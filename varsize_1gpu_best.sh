# Copyright (c) 2023-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
#!/bin/bash

#SBATCH --nodelist= <node list>
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --output= <output file name>

hostname
whoami
echo $CUDA_VISIBLE_DEVICES
source .bashrc # or .zsh or whatever source file you use
cd <directory of code>
conda activate <name of environment>
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port 53533 --use_env main.py --model deit_small_patch16_224 --batch-size 64 --data-path <path to imagenet> --output_dir <output directory> --lr 1e-3 --localvit --localvit-act 'hs' --is-multisize --init-size 32 --epoch-step 5 --patch-step 2 --eval-on-final-size
echo "DONE"
