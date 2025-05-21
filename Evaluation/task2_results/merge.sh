#!/bin/bash

# Directory containing files to merge
input_dir="/home/bruno/MRGC/SLAM/L3/Mini-SLAM/task2_results"
# Output file
output_file="merged.txt"

# Clear or create the output file
> "$output_file"

# Loop over files in sorted order
for file in $(ls -1v "$input_dir"); do
  # Check if it is a regular file (ignore directories)
  if [ -f "$input_dir/$file" ]; then
    cat "$input_dir/$file" >> "$output_file"
  fi
done

echo "Merged files into $output_file"
