#!/bin/bash

for i in {1..10}; do
    ./Apps/monoTUMIV /home/bruno/MRGC/SLAM/L3/Mini-SLAM/Data/dataset-room1_512_16 /home/bruno/MRGC/SLAM/L3/Mini-SLAM/Apps/TUM_TimeStamps/dataset-room1_512.txt  
    mv trajectory.txt "output_${i}.txt"
done

