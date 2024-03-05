# Problem Set 3 Corrected (March 2024)
# SFU ECON832 Computational Methods for Economics
# Spring 2024

# Problem 2: This script develops the initial supply and demand model, and develops the functions we need available in all processors/workers for parallelization. 
# Source: https://juliaeconomics.com/2014/06/18/parallel-processing-in-julia-bootstrapping-the-mle/

# Preliminaries ---------------------------------------------------------------

# Import packages

import JuMP 
import Ipopt 
using Random 
using DataFrames
using Distributions 
using Distributed
using Statistics

# Declaring the supply and demand model  -----------------------------------------------------

## Parameters ------------------------------------------------------------------

Random.seed!(78909434) 

N = 100
a = 0.5 # Demand intercept
b = 0.5 # Demand slope (note that we define as positive because in the demand equation there will be a negative sign attached to it)
α = 0.5 # Supply intercept
β = 0.5 # Supply slope

σ_u = 1.5 # Std of demand shocks
σ_v = 2.5 # Std of supply shocks

μ_u = 0.0 # Mean of demand shocks
μ_v = 0.0 # Mean of supply shocks

## Simulation of observational data --------------------------------------------

# u and v (random supply and demand shocks)

u = zeros(N)
v = zeros(N) 

for i in 1:N
    u[i] = rand(Normal(μ_u, σ_u))
    v[i] = rand(Normal(μ_v, σ_v))
end

# price vector p

p = zeros(N)

for i in 1:N
    p[i]=((a-α)/(β+b)) + ((u[i]-v[i])/(β+b))
end

# quantity in the market y

y = zeros(N)

for i in 1:N
    y[i]= ((a*β + b*α)/(b+β)) + ((β*u[i] + b*v[i])/(β+b))
end

# D and S vectors (quantities)

D = zeros(N)
S = zeros(N)

for i in 1:N
    D[i] = a - b*p[i] + u[i] 
    S[i] = α + β*p[i] + v[i]
end

# Relationships of instruments with random shocks

c_u = 0.5
c_v = 0.5

# Standard deviations of the unobserved parts of random shocks

σ_ϵu = 0.25
σ_ϵv = 0.5

# Means of the unobserved parts of random shocks

μ_ϵ_u = 0.0
μ_ϵ_v = 0.0

# Error of the instruments

ϵ_u = zeros(N)
ϵ_v = zeros(N)

for i in 1:N
    ϵ_u[i] = rand(Normal(μ_ϵ_u, σ_ϵu)) 
    ϵ_v[i] = rand(Normal(μ_ϵ_v, σ_ϵv))
end 

x_u = zeros(N)
x_v = zeros(N)

for i in 1:N
    x_u[i] = (( u[i] - ϵ_u[i] ) / (c_u) )
    x_v[i] = (( v[i] - ϵ_v[i] ) / (c_v)) 
end

# 2SLS estimation & standard errors -------------------------------------------------------------

## Parameters ------------------------------------------------------------------

# I will need a dataframe with the data that I have (my sample) to perform the bootstrapping

df = DataFrame(p = p, y = y, x_v = x_v)

## Without parallelization --------------------------------------------------------

function two_stage_least_squares(df)
    N = 100
        # First stage
        first_stage = JuMP.Model(Ipopt.Optimizer)
        JuMP.@variable(first_stage, π0) # Pi 0 is the intercept of the first stage regression
        JuMP.@variable(first_stage, π1) # Pi 1 is the slope of the first stage regression
        JuMP.@objective(first_stage, Min, sum((df.p[i] - π0 - π1*df.x_v[i])^2  for i in 1:N))
        JuMP.optimize!(first_stage) # Perform the optimization
        π0_hat = JuMP.value.(π0) # Access the intercept
        π1_hat = JuMP.value.(π1) # Access the slope
        p_hat = zeros(N) # Create a vector of zeros to later allocate values
        for i in 1:N
            p_hat[i] = π0_hat + π1_hat*df.x_v[i]
        end
        # Second stage
        second_stage = JuMP.Model(Ipopt.Optimizer)
        JuMP.@variable(second_stage, γ0) # Gamma 0 is the intercept of the second stage regression
        JuMP.@variable(second_stage, γ1) # Gamma 1 is the slope of the second stage regression
        JuMP.@objective(second_stage, Min, sum((df.y[i] - γ0 - γ1*p_hat[i])^2  for i in 1:N)) # Minimize the sum of squared residuals. This is the objective function
        JuMP.optimize!(second_stage) # Perform the optimization
        γ1_hat = -JuMP.value.(γ1) # Access the slope of the second stage regression
        return γ1_hat
end

# try the function on df

two_stage_least_squares(df) # output is \hat{β}

# Function to perform the bootstrapping on the model

function bootstrap_sample(df)
    sample = df[rand(1:size(df, 1), size(df, 1)),:]  # Bootstrap sample
    return two_stage_least_squares(sample)
end

# Apply the function to the model with pmap (takes a very very long time to run)

# β_boot = pmap(bootstrap_sample, [df for i in 1:B]) 

# Compute the β/b ratio 

# β_ratio = β_boot/b

# bootstrap_se = std(β_ratio)

# println("The bootstrap standard error for β/b is ", bootstrap_se)