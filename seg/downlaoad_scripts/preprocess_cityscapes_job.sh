#!/bin/bash -l                     
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=1:00:00
#SBATCH --job-name=preprocess_cityspaces
#SBATCH --export=NONE
                                   
unset SLURM_EXPORT_ENV 

cd $HOME/sim2real/repos/MIC/seg

module load python/3.8-anaconda
module load cuda/11.1.0
conda activate mic-seg

python tools/convert_datasets/cityscapes.py data/cityscapes --nproc 8