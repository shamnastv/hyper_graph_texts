#!/bin/sh
#SBATCH --job-name=Ablation # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=47:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 main.py --num_layers 2 --num_mlp_layers 1 --dataset ohsumed --seed -1 --hidden_dim 50
python3 main.py --num_layers 4 --num_mlp_layers 1 --dataset ohsumed --seed -1 --hidden_dim 50
python3 main.py --num_layers 5 --num_mlp_layers 1 --dataset ohsumed --seed -1 --hidden_dim 50

