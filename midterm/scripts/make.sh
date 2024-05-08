#!/bin/bash
# Make file to run all Julia files in the current directory
# Run in windows with Git Bash
# Loop over all Julia files in the scripts directory of the repository

for file in scripts/*.jl
do
  # Run the Julia file
  julia "$file"
done

# Optionally, compile the Quarto report (only if Latex is installed)

# quarto preview report.qmd