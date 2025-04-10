#!/bin/bash

# Submit the server job
echo "Submitting RAG server job..."
SERVER_JOB_ID=$(sbatch run_rag_service.slurm | awk '{print $4}')

if [ -z "$SERVER_JOB_ID" ]; then
    echo "Failed to submit server job"
    exit 1
fi

echo "RAG server job submitted with ID: $SERVER_JOB_ID"

# Create a temporary client script with the server job ID
echo "Creating client job script with dependency on server job..."
sed "s/afterok:JOBID/afterok:$SERVER_JOB_ID/" run_rag_client.slurm > run_rag_client_temp.slurm

# Submit the client job with dependency on the server job
echo "Submitting RAG client job..."
CLIENT_JOB_ID=$(sbatch run_rag_client_temp.slurm | awk '{print $4}')

if [ -z "$CLIENT_JOB_ID" ]; then
    echo "Failed to submit client job"
    exit 1
fi

echo "RAG client job submitted with ID: $CLIENT_JOB_ID"

# Clean up temporary script
rm run_rag_client_temp.slurm

echo ""
echo "Jobs submitted successfully. Check status with: squeue -u $USER"
echo "Server log: rag_service_${SERVER_JOB_ID}.log"
echo "Client log: rag_client_${CLIENT_JOB_ID}.log" 