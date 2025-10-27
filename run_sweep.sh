#!/bin/bash

# --- Configuration ---
# Set the parameters for your experiment sweep
ENVIRONMENT="tic_tac_toe"
OBS_TYPE="txt"
AGENT="o1mini_agent"
MODEL="o1-mini"

# List of demonstration numbers to run (matches the paper's log scale)
DEMONSTRATIONS=(0 2 4 8 16 32 64 128 256)

# --- Run Loop ---
echo "Starting experiment sweep for $MODEL on $ENVIRONMENT"

for demos in "${DEMONSTRATIONS[@]}"
do
  echo "--- Running with $demos demonstration(s) ---"
  python -m src.main \
    --environment="$ENVIRONMENT" \
    --observation_type="$OBS_TYPE" \
    --agent="$AGENT" \
    --model_name="$MODEL" \
    --num_demonstrations="$demos"
done

echo "Sweep complete!"