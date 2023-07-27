#!/bin/bash

for i in {1..26}; do
    echo "Executing optimize.py for $i"
    python optimize.py --state="$1" --variable="$2"
done