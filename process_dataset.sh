#!/usr/bin/env bash

# Always run relative to the folder containing this script.
cd "$(dirname "$0")" || exit 1

for folder in train_karate_detection/dataset/*/; do
    echo "========================================"
    echo "Processing: $folder"
    echo "========================================"

    python3 demo_webcam.py --input "$folder" --save

    if [ $? -ne 0 ]; then
        echo "ERROR: Failed while processing $folder"
        exit 1
    fi
done

echo "All dataset folders were processed successfully."
