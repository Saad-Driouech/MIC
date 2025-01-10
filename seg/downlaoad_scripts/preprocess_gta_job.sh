#!/bin/bash -l                     
#
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=2:00:00
#SBATCH --job-name=preprocess_gta
#SBATCH --export=NONE
                                   
unset SLURM_EXPORT_ENV 

cd $HOME/sim2real/repos/MIC/seg    
ln -s /home/woody/iwnt/iwnt134h/MIC/data $HOME/sim2real/repos/MIC/seg/data

module load python/3.8-anaconda
module load cuda/11.1.0
conda activate mic-seg

python tools/convert_datasets/gta.py data/gta --nproc 8