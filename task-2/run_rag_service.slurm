#!/bin/bash
#SBATCH --job-name=rag_service
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --output=rag_service_%j.log

# Load modules (adjust as needed for your environment)
module load python/3.10 cuda/11.7

# Activate your conda environment if it exists
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag

# Navigate to the directory with your code
cd $SLURM_SUBMIT_DIR/task-2

# Install requirements if needed
pip install -r requirements.txt

# Print GPU information
nvidia-smi

# Run the server
python serving_rag.py 