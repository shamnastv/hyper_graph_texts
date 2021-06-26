#!/bin/sh
#SBATCH --job-name=R8 # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

#python3 main.py --num_layers 3 --dataset R8 --seed -1 --num_clusters 2 --early_stop 10 --weight_decay 1e-7 --lr 1e-3 --num_exp 2 --dropout .5
#python3 main.py --num_layers 3 --dataset R8 --seed -1 --num_clusters 2 --early_stop 10 --weight_decay 1e-7 --lr 5e-3 --num_exp 2 --dropout .5

#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset R8 --lda
#python3 main.py --lr 5e-4 --hidden_dim 200 --num_layers 3 --dropout .3 --dataset R8 --num_exp 2 --num_clusters 0
#python3 main.py --lr 5e-4 --hidden_dim 200 --num_layers 3 --dropout .3 --dataset R8 --num_exp 2 --num_clusters 1
python3 main.py --lr 3e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --dataset R8 --num_exp 2 --num_clusters 2
python3 main.py --lr 3e-3 --hidden_dim 200 --num_layers 3 --dropout .5 --dataset R8 --num_exp 2 --num_clusters 2