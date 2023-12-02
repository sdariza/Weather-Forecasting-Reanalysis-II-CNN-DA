    #!/bin/bash

for i in {0,6,12,18}; do
    echo "Executing train models for state $i"
    python data_driven/train_cnn_models.py --state="$i" --variable="air"
    python data_driven/train_cnn_models.py --state="$i" --variable="vwnd"
    python data_driven/train_cnn_models.py --state="$i" --variable="uwnd"
done