#!/bin/bash

for i in {0,6,12,18}; do
    echo "Executing train models for state $i"
    python train_cnn_models.py --state="$i" --variable="air"
    python train_cnn_models.py --state="$i" --variable="vwnd"
    python train_cnn_models.py --state="$i" --variable="uwnd"
done