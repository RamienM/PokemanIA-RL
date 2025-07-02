#!/bin/bash

echo "Pokémon Red Training - Automatic Restart with Dynamic Checkpoint"
echo "Starting training loop..."

while true; do
    echo "--------------------------"
    echo "Checking for previous checkpoint..."
    echo "--------------------------"

    CHECKPOINT_PATH=""

    # If the checkpoint path file exists, read its content
    if [[ -f "last_checkpoint_path.txt" ]]; then
        CHECKPOINT_PATH=$(<last_checkpoint_path.txt)
        echo "Checkpoint found: $CHECKPOINT_PATH"
    else
        echo "No previous checkpoint found."
    fi

    echo "--------------------------"
    echo "Running training..."
    echo "--------------------------"

    # Run the training script, passing checkpoint path if it exists
    if [[ -n "$CHECKPOINT_PATH" ]]; then
        python train.py "$CHECKPOINT_PATH"
    else
        python train.py
    fi

    echo
    echo "Training finished. Restarting in 5 seconds..."
    sleep 5
done
