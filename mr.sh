#!/bin/sh
#SBATCH --job-name=MR # Job name
#SBATCH --ntasks=4 # Run on a single CPU
#SBATCH --time=23:50:00 # Time limit hrs:min:sec
#SBATCH --output=test_job%j.out # Standard output and error log
#SBATCH --gres=gpu:1
#SBATCH --partition=cl1_48h-1G

python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 1 --early_stop 10 --lr 5e-4  --weight_decay 1e-4
python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 1 --early_stop 10 --lr 5e-3  --weight_decay 1e-4

python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 1 --early_stop 10 --lr 5e-4  --weight_decay 1e-7
python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 1 --early_stop 10 --lr 5e-3  --weight_decay 1e-7

python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 1 --early_stop 10 --lr 5e-4  --weight_decay 1e-12
python3 main.py --num_layers 3 --dataset mr --seed -1 --num_clusters 1 --early_stop 10 --lr 5e-3  --weight_decay 1e-12

#python3 main.py --lr 5e-3 --num_layers 3 --num_mlp_layers 1 --dataset mr --seed -1 --num_clusters 1
#python3 main.py --lr 5e-4 --num_layers 3 --num_mlp_layers 1 --dataset mr --seed -1 --num_clusters 1

#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset mr --lda
#python3 main.py --lr 1e-3 --hidden_dim 200 --num_layers 3 --dropout .3 --num_mlp_layers 1 --dataset mr --random_vec
