#!/bin/bash

#SBATCH --job-name=Show_Results
#SBATCH --nodes=1 --ntasks-per-node=6 --gres=gpu:H100:1
#SBATCH --time=00:05:00
#SBATCH --output=./logs/show_results.out
#SBATCH --mail-type=NONE

cd $HOME/Senior_Design/MotionIndustries-ImageToProduct/Model_Development/Training_Scripts || exit

module load anaconda3
conda activate Senior_Design
python3 visualize_model.py