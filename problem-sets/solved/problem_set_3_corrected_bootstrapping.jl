# Problem Set 3 Corrected (March 2024)
# SFU ECON832 Computational Methods for Economics
# Spring 2024

# Problem 2: This script develops the initial supply and demand model, and develops the functions we need available in all processors/workers for parallelization. 
# Source: https://juliaeconomics.com/2014/06/18/parallel-processing-in-julia-bootstrapping-the-mle/

# Use bootstrapping to provide boostrap standard errors for β/b in the supply and demand model.
# Import the bootstrap functions from the file problem_set_3_bootstrap_functions.jl

# Preliminaries ---------------------------------------------------------------

# Import packages

using Distributed
using Optim
using DataFrames

# Bootstrapping procedure -------------------------------------------------------

addprocs(4)

@everywhere include("problem_set_3_bootstrap_functions.jl")

B = 10000 # Total number of bootstrap samples

# Apply the bootstrap_sample function to each worker

beta_samples_pmap = pmap(bootstrap_sample, [df for _ in 1:B])

# Divide by b to get the bootstrap samples for β/b

beta_samples_pmap = beta_samples_pmap ./ 0.5

# Calculate the standard errors

std_errors = std(beta_samples_pmap)

# Print the bootstrap standard error

println("The bootstrap standard error for β/b is:", std_errors)