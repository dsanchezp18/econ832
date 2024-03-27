# Problem Set 3
# SFU ECON832 Computational Methods for Economics
# Spring 2024

# Problem 1: Utility Calculations

# Preliminaries --------------------------------------------------------------

# Load Packages

using DataFrames
using CSV
using Statistics

# Load data for the lotteries

lotteries = CSV.read("problem-sets/pset3_rulc_replication/data/lotteries.csv", DataFrame)

# Calculating CRRA Utility for Table 1 Rankings ------------------------------

# Change the name of Expectation to X in the dataframe

rename!(lotteries, :Expectation => :X)

# Define Table 1 sigma values

σ = [-2, 0, 0.25, 0.3, 0.5, 0.75]

# Calculate CRRA utility for each sigma value, and add to the dataframe. U(X) = X^(1-σ)/(1-σ). 

for i in 1:length(σ)
    lotteries[!, Symbol("U_$(σ[i])")] = (lotteries.X).^(1-σ[i])/(1-σ[i])
end

# Create a ranking of the first utility column, sorting it in descending order. 1 is the highest utility, 6 is the lowest.

lotteries[!, :Ranking] = sortperm(lotteries[!, Symbol("U_$(σ[1])")], rev=true)

# Do the same for all other utility columns

for i in 2:length(σ)
    lotteries[!, Symbol("Ranking_$(σ[i])")] = sortperm(lotteries[!, Symbol("U_$(σ[i])")], rev=true)
end

# Repeat for the following vector of sigma values

σ_modified = [-1, 0.22, 0.26, 0.27, 0.28, 0.30, 1]

# Calculate, using logarithm for the cases where it applies, and the fraction otherwise

for i in 1:length(σ_modified)
    if σ_modified[i] == 0
        lotteries[!, Symbol("U_$(σ_modified[i])")] = log.(lotteries.X)
    else
        lotteries[!, Symbol("U_$(σ_modified[i])")] = (lotteries.X).^(1-σ_modified[i])/(1-σ_modified[i])
    end
end

# Create a ranking of the first utility column, sorting it in descending order. 1 is the highest utility, 6 is the lowest.

lotteries[!, :Ranking] = sortperm(lotteries[!, Symbol("U_$(σ_modified[1])")], rev=true)

# Do the same for all other utility columns

for i in 2:length(σ_modified)
    lotteries[!, Symbol("Ranking_$(σ_modified[i])")] = sortperm(lotteries[!, Symbol("U_$(σ_modified[i])")], rev=true)
end

# Save the dataframe to a csv file

CSV.write("problem-sets/pset3_rulc_replication/data/lotteries_with_utilities.csv", lotteries)