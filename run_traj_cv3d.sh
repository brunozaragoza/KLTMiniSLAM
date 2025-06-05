#!/bin/bash


./Apps/monoCV3D ./Data/cecum_t1_a/ ./Data/CV3D.yaml  
mv trajectory.txt "output_${i}.txt"


