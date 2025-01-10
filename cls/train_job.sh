#!/bin/bash -l                     
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=08:00:00
#SBATCH --job-name=train_ViT_VisDA
#SBATCH --export=NONE
                                   
unset SLURM_EXPORT_ENV 

export PYTHONWARNINGS="ignore::DeprecationWarning"

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

cd $HOME/sim2real/repos/MIC/cls

module load python/3.8-anaconda
conda activate mic-cls

python -W ignore::DeprecationWarning run_experiments.py --exp 1