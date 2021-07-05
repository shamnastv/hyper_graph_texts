#!/bin/sh
#SBATCH --job-name=MR # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

#python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 1 --early_stop 10 --lr 5e-4  --weight_decay 1e-6


python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 0 --num_exp 4 --early_stop 10 --lr 8e-4 --dropout .5
python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 1 --num_exp 4 --early_stop 10 --lr 8e-4 --dropout .5
python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 2 --num_exp 4 --early_stop 10 --lr 8e-4 --dropout .5

#python3 main.py --lr 5e-3 --num_layers 3 --num_mlp_layers 1 --dataset mr --seed -1 --num_clusters 1
#python3 main.py --lr 5e-4 --num_layers 3 --num_mlp_layers 1 --dataset mr --seed -1 --num_clusters 1

#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset mr --lda
#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset mr --random_vec
