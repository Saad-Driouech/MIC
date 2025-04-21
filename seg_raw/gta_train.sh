#!/bin/bash -l                     
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=24:00:00
#SBATCH --job-name=job_gta_trafo_train
#SBATCH --export=NONE
                                   
unset SLURM_EXPORT_ENV 

export http_proxy=http://proxy:80
export https_proxy=http://proxy:80

cd $HOME/sim2real/repos/MIC/seg_raw

module load python/3.10-anaconda
conda activate seg_raw

python train.py --image_dir /home/hpc/iwnt/iwnt134h/sim2real/repos/MIC/seg/data/gta/generated --label_dir /home/hpc/iwnt/iwnt134h/sim2real/repos/MIC/seg/data/gta/labels
