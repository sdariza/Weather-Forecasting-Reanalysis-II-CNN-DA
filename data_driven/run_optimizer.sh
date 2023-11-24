#!/bin/bash

for i in {1..150}; do
    echo "Executing optimize.py for $i"
    python data_driven/optimize.py --state="$1" --variable="air"
    python data_driven/optimize.py --state="$1" --variable="vwnd"
    python data_driven/optimize.py --state="$1" --variable="uwnd"
done