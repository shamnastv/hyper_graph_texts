#!/bin/sh
#SBATCH --job-name=OHSUMED # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .5 --num_mlp_layers 1 --dataset ohsumed
#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset ohsumed --lda
#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset ohsumed --random_vec

python3 main.py --num_layers 3 --num_mlp_layers 1 --dataset ohsumed --seed -1 --hidden_dim 50 --num_exp 2 --early_stop 15
python3 main.py --num_layers 3 --num_mlp_layers 1 --dataset ohsumed --seed -1 --hidden_dim 100 --num_exp 2 --early_stop 15
python3 main.py --num_layers 3 --num_mlp_layers 1 --dataset ohsumed --seed -1 --hidden_dim 300 --num_exp 2 --early_stop 15

