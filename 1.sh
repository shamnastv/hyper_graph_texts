#!/bin/sh
#SBATCH --job-name=Ablation # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=47:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 main.py --num_layers 3 --dataset ohsumed --seed -1 --num_clusters 3 --num_exp 2 --early_stop 15 --val_prop .25
python3 main.py --num_layers 3 --dataset ohsumed --seed -1 --num_clusters 3 --num_exp 2 --early_stop 15 --val_prop .5
python3 main.py --num_layers 3 --dataset ohsumed --seed -1 --num_clusters 3 --num_exp 2 --early_stop 15 --val_prop .75
python3 main.py --num_layers 3 --dataset ohsumed --seed -1 --num_clusters 3 --num_exp 2 --early_stop 15 --val_prop .95
