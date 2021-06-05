#!/bin/sh
#SBATCH --job-name=R8 # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

#python3 main.py --hidden_dim 200 --num_layers 3 --num_mlp_layers 1 --dataset R8 --seed -1
python3 main.py --hidden_dim 200 --num_layers 3 --num_mlp_layers 1 --dataset R8 --seed -1 --lr 1e-3
python3 main.py --hidden_dim 200 --num_layers 3 --num_mlp_layers 1 --dataset R8 --seed -1 --lr 5e-4
python3 main.py --hidden_dim 200 --num_layers 3 --num_mlp_layers 1 --dataset R8 --seed -1 --lr 1e-4
python3 main.py --hidden_dim 200 --num_layers 3 --num_mlp_layers 1 --dataset R8 --seed -1 --lr 5e-5


#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset R8 --lda
#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset R8 --random_vec
