#!/bin/bash -l                     
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=download_gta_images
#SBATCH --export=NONE
                                   
unset SLURM_EXPORT_ENV 

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

cd $HOME/sim2real/repos/MIC

srun ./download_gta_images.sh