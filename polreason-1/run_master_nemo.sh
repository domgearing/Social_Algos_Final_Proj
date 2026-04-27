#!/bin/bash

# ==============================================================================
# Slurm Submission Script for Polreason Analysis Pipeline (Nemo Subset)
# ==============================================================================
#
# This script executes the 'master.R' analysis pipeline for specific Nemo models.
#
# Usage:
#   sbatch run_master_nemo.sh
#
# ==============================================================================

#SBATCH --job-name=polreason_nemo
#SBATCH --output=logs/nemo_%j.out
#SBATCH --error=logs/nemo_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=pi_ju78
#SBATCH --partition=day_amd

# --- Environment Setup ---

# Create logs directory if it doesn't exist
mkdir -p logs

# Load R module (Adjust module name based on your cluster's naming convention)
# Some clusters use 'R/4.3.1', others just 'R'.
module load R || module load R/4.3.3 || echo "Warning: R module not found. Assuming R is in PATH."

# --- Execute Analysis ---

echo "Starting polreason analysis pipeline: master.R (Nemo Subset)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODENAME"
echo "Start time: $(date)"

# Run the master R script with specific rater names as arguments
# Note: master.R has been updated to accept these as filters.
# Rscript analysis/scripts/master.R "mistralai_mistral-nemo_OLD" "Nemo_2" "Nemo_v3" "nemo_4" "nemo_5" "Dom Nemo v2"
Rscript analysis/scripts/master.R "Dom Nemo v2"

echo "Pipeline completed at: $(date)"
