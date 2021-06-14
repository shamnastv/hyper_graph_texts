#!/bin/sh
#SBATCH --job-name=Ablation # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=47:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G
python3 main.py --num_layers 3 --dataset R8 --seed -1 --num_clusters 2 --early_stop 10 --weight_decay 1e-7 --lr 1e-2 --num_exp 2 --dropout .5

#python3 main.py --num_layers 3 --dataset 20ng --seed -1 --num_clusters 1 --num_exp 4

