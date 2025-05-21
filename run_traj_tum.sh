#!/bin/bash

for i in {1..10}; do
    ./Apps/mono_tumrgbd Data/rgbd_dataset_freiburg1_xyz/ ./Data/TUMRGBD.yaml  
    mv trajectory.txt "output_${i}.txt"
done

