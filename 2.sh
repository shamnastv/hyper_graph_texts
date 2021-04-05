#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=47:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 4 --dropout .4 --num_mlp_layers 2 --dataset R8 --early_stop 50
python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 5 --dropout .4 --num_mlp_layers 2 --dataset R8 --early_stop 50
python3 main.py --lr 1e-3 --hidden_dim 100 --num_layers 4 --dropout .4 --num_mlp_layers 2 --dataset R8 --early_stop 50
python3 main.py --lr 1e-3 --hidden_dim 100 --num_layers 5 --dropout .4 --num_mlp_layers 2 --dataset R8 --early_stop 50
