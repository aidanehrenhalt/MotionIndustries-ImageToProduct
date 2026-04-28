#!/bin/bash

#SBATCH --job-name=Five_Conv_Training_Focal_Loss
#SBATCH --nodes=1 --ntasks-per-node=24 --gres=gpu:H100:1
#SBATCH --time=2:00:00
#SBATCH --output=./logs/five_conv_training_focal_loss.out
#SBATCH --mail-type=NONE

cd $HOME/Senior_Design/MotionIndustries-ImageToProduct/Model_Development/Training_Scripts || exit

module load anaconda3
conda activate Senior_Design
python3 five_conv_focal.py