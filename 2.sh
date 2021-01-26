#!/bin/sh
#SBATCH --job-name=GNN # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 main.py --lr 5e-4 --hidden_dim 200 --num_layers 3 --dropout .4 --num_mlp_layers 1 --lda
python3 main.py --lr 5e-4 --hidden_dim 200 --num_layers 3 --dropout .4 --num_mlp_layers 1 --lda --dataset R8
