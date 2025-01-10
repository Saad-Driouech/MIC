#!/bin/bash -l                     
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=train_seg_DAFormer_CAugCS
#SBATCH --export=NONE
                                   
unset SLURM_EXPORT_ENV 

export PYTHONWARNINGS="ignore::DeprecationWarning"

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

cd $HOME/sim2real/repos/MIC/seg

module load python/3.8-anaconda
module load cuda/11.1.0
conda activate mic-seg

python -W ignore::DeprecationWarning run_experiments.py --exp 81