#!/bin/bash

# --- Configuration ---
# Set the parameters for your experiment sweep
ENVIRONMENT="tic_tac_toe"
OBS_TYPE="txt"
AGENT="llama_agent"
MODEL="llama3"

# Set the wandb project name
WANDB_PROJECT="lm-act-tic-tac-toe"
# Set the run name prefix
RUN_PREFIX="llama_basic_sweep"

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