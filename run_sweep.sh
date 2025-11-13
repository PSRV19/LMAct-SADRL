#!/bin/bash

# --- Configuration ---
# Set the parameters for your experiment sweep
ENVIRONMENT="tic_tac_toe"
OBS_TYPE="txt"
AGENT="gpt4o_agent"
MODEL="gpt-4o"
RUN_NAME="exp3"

# List of demonstration numbers to run (matches the paper's log scale)
DEMONSTRATIONS=(4 8 16 32 64 128 256)

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
    --num_demonstrations="$demos" \
    --run_name_prefix="$RUN_NAME"
done

echo "Sweep complete!"