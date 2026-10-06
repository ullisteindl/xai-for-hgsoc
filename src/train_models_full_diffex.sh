#!/bin/bash
#SBATCH --job-name=hp_tune_full_diffex
#SBATCH --partition=gpu_node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=15G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/hp_tune_full_diffex_%A_%a.out
#SBATCH --error=logs/hp_tune_full_diffex_%A_%a.err
#SBATCH --array=0-4

# 1. Define your list of SIDs (Order must match the array indices above)
SIDS=("original_control" "only_treated" "PID_final" "cell_type_simplified" "condition")

# 2. Get the current SID based on this specific task's array index
sid=${SIDS[$SLURM_ARRAY_TASK_ID]}

echo "Starting job for index ${SLURM_ARRAY_TASK_ID}, SID: ${sid}"
mkdir -p logs

# 3. Use srun inside the allocation to execute the Python script on the GPU
eval "$(mamba shell hook --shell bash)"
mamba activate deeltorchlip
export TQDM_DISABLE=True
srun python hp_tune_deel_torchlip.py \
    --data ../data/processed/${sid}_full_train_dataset_logcounts_features.npy \
    --architecture deel-torchlib \
    --trials 50 \
    --max_epochs 200 \
    --project_name hp_tuning_${sid}_diffex_full \
    --batch_center \
    --diffex


