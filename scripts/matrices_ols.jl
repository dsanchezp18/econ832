# Comparing OLS with regressions and OLS

using DataFrames
using CSV
using GLM
using TidierData
using LinearAlgebra
using Distributions
using Random

# Simulate data for a regression

# Set seed for reproducibility

Random.seed!(123)

n = 100

X1 = randn(n)

X2 = randn(n)

β1 = 0.5

β2 = 0.3

β0 = 1

ϵ = randn(n)

y = β0 .+ β1*X1 .+ β2*X2 .+ ϵ

# Run OLS regression using matrices

X = hcat(ones(n), X1, X2)

β = inv(X'X)*(X'y)

# Run OLS regression using GLM

df = DataFrame(y = y, X1 = X1, X2 = X2)

ols = lm(@formula(y ~ X1 + X2), df)