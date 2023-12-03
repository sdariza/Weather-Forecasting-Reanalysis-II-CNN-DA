#!/bin/bash

for i in {1..200}; do
    echo "Executing optimize.py for $i"
    python optimize.py --state="$1" --variable="air"
    python optimize.py --state="$1" --variable="vwnd"
    python optimize.py --state="$1" --variable="uwnd"
done