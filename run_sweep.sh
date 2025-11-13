#!/bin/bash

# --- Configuration ---
# Set the parameters for your experiment sweep
ENVIRONMENT="grid_world"
OBS_TYPE="txt"
AGENT="gpt4o_agent"
MODEL="gpt-4o"

# Set the wandb project name
WANDB_PROJECT="lm-act-grid-world"
# Set the run name prefix
RUN_PREFIX="basic_curriculum_grid_world_"

# List of demonstration numbers to run
DEMONSTRATIONS=(0 2 4 8 16 32 64 128 256)

# --- Run Loop ---
echo "Starting experiment sweep for $MODEL on $ENVIRONMENT"
echo "Logging to wandb project: $WANDB_PROJECT"

for demos in "${DEMONSTRATIONS[@]}"
do
  echo "--- Running with $demos demonstration(s) ---"
  python -m src.main \
    --environment="$ENVIRONMENT" \
    --observation_type="$OBS_TYPE" \
    --agent="$AGENT" \
    --model_name="$MODEL" \
    --num_demonstrations="$demos" \
    --wandb_project="$WANDB_PROJECT" \
    --run_name_prefix="$RUN_PREFIX"
done

echo "Sweep complete!"