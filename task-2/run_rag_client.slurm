#!/bin/bash
#SBATCH --job-name=rag_client
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=rag_client_%j.log
#SBATCH --dependency=afterok:JOBID  # Replace JOBID with the job ID of the server job

# Load modules (adjust as needed for your environment)
module load python/3.10

# Activate your conda environment if it exists
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rag

# Navigate to the directory with your code
cd $SLURM_SUBMIT_DIR/task-2

# Install requests if needed
pip install requests

# Wait a moment to ensure the server is fully started
sleep 20

# Get the hostname of the node running the server job
# This assumes you're running both jobs on the same cluster
SERVER_HOST=$(squeue --job=$SLURM_DEPENDENCY_AFTER_OK --noheader --format=%N)

# If server hostname can't be determined, default to localhost
if [ -z "$SERVER_HOST" ]; then
    SERVER_HOST="localhost"
    echo "Couldn't determine server hostname, using localhost"
else
    echo "Server is running on $SERVER_HOST"
fi

# Run the client with the server hostname
python test_rag_client.py --host $SERVER_HOST --port 8000 --parallel 10 