#!/bin/bash
#SBATCH --job-name=panda-train
#SBATCH --error=panda-train.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --partition="gpu"
#SBATCH --time=2-12:00:00
#SBATCH --mem=40G
#SBATCH -A raiselab-paid




conda_env_name=mpd
conda_env_path=/path/to/dir/.conda/envs/$conda_env_name


module purge
module load ffmpeg
module load gcc
module load ninja
module load anaconda
source activate $conda_env_name
export PATH="$conda_env_path/bin:$PATH"
export LD_LIBRARY_PATH="$conda_env_path/lib:$LD_LIBRARY_PATH"

# python train_score_model.py
python train_sb_ps3d.py
