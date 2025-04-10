#!/bin/bash
#SBATCH --job-name=rag_service
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=rag_service_%j.log

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag

# Navigate to the directory with your code
cd $SLURM_SUBMIT_DIR

# Print GPU information
nvidia-smi

# Run the server
python serving_rag_original.py 