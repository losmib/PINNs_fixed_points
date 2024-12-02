#!/bin/bash

# Input file
input_file="setup.yaml"

# Initialize the counter
counter=0

# Outer loop: simulation time
for T in $(seq 1 1 10); do
    
    # Inner loop: initial condition
    for y0 in $(LC_NUMERIC=C seq 0.05 0.1 1.3); do
        
	# Increment the counter
        counter=$((counter + 1))

    	output_file="job_${counter}.yaml"

    	# Replace the values in the output file
	sed "s|version: 'vanilla/default'|version: 'vanilla/T-${T}_y0-${y0}'|; s|T: [0-9]\+|T: ${T}|; s|y0: [0-9]\+|y0: ${y0}|" "$input_file" > "$output_file"

	# Print the current values of T, y0, and counter
        echo "T=$T, y0=$y0, Job_ID=$counter"

    done
done


