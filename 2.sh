#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=47:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 main.py --lr 1e-3 --hidden_dim 50 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset R8 --early_stop 20
python3 main.py --lr 1e-3 --hidden_dim 100 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset R8 --early_stop 20
python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset R8 --early_stop 20
python3 main.py --lr 1e-3 --hidden_dim 300 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset R8 --early_stop 20
