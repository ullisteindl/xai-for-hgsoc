#!/bin/bash
#SBATCH --job-name=model_attr
#SBATCH --output=logs/attr_%A_%a.out
#SBATCH --error=logs/attr_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpu_node
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --array=0-4

eval "$(mamba shell hook --shell bash)"
mamba activate deeltorchlip
export TQDM_DISABLE=True

# 1. Define the array of variables to substitute for "original_control"
# Add, remove, or modify these names to match your actual dataset variations
CONDITIONS=("original_control" "only_treated" "PID_final" "cell_type_simplified" "condition")

# 2. Get the condition corresponding to this specific task index
CURRENT_VAR=${CONDITIONS[$SLURM_ARRAY_TASK_ID]}

echo "Starting Slurm Task ID: $SLURM_ARRAY_TASK_ID"
echo "Processing dataset variable: $CURRENT_VAR"


# 4. Run the Python script using the dynamic variable
python calculate_model_attributions.py \
    --model-path "hp_tuning_${CURRENT_VAR}_full/best_model.pt" \
    --data "../data/processed/${CURRENT_VAR}_full_train_dataset_logcounts_features.npy" \
    --hidden-layers "64,32,16" \
    --batch-size 32 \
    --architecture='deel-torchlib' \
    --batch-center

