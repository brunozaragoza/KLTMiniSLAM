#!/bin/bash

for i in {1..10}; do
    python3 evaluate_ate_scale.py "GroundTruth/fr1_xyz.txt" "task_4_data/output_${i}.txt" --plot "task_4_data/output_${i}.png"
done

