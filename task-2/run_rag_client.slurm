#!/bin/bash
#SBATCH --job-name=rag_client
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=rag_client_%j.log
#SBATCH --dependency=afterok:JOBID  # Replace JOBID with the job ID of the server job

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag

# Navigate to the directory with your code
cd $SLURM_SUBMIT_DIR

# Install requests if needed
pip install requests

# Wait for server to fully start
sleep 30

# Get the hostname of the node running the server job
SERVER_NODE=$(squeue --job=$SLURM_DEPENDENCY_AFTER_OK --noheader --format=%N)
echo "Server node: $SERVER_NODE"

# Run the client
echo "Starting client, connecting to server at $SERVER_NODE:8000"
python test_rag_client.py --host $SERVER_NODE --port 8000 --parallel 5 