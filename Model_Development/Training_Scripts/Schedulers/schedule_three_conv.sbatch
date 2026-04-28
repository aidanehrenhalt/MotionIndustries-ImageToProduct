#!/bin/bash

#SBATCH --job-name=Three_Conv_Training
#SBATCH --nodes=1 --ntasks-per-node=24 --gres=gpu:H100:1
#SBATCH --time=2:00:00
#SBATCH --output=./logs/three_conv_training.out
#SBATCH --mail-type=NONE

cd $HOME/Senior_Design/MotionIndustries-ImageToProduct/Model_Development/Training_Scripts || exit

module load anaconda3
conda activate Senior_Design
python3 three_conv_train.py