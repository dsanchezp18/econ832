# Assignment 1 Instrumental Variables for Supply & Demand
# Daniel Sanchez
# ECON832 Computational Methods for Economics
# Spring 2024

# -------------------------------------------------------------------------- #
# Preliminaries
# -------------------------------------------------------------------------- #

# Import packages (they need to be installed in the computer with using Pkg, Pkg.add("Package Name")

import JuMP # optimizer
import Ipopt # optimizer

using Random # For random numbers
using Distributions # For probability distributions

# Pre define parameters to be used below 

Random.seed!(78909434) # for reproducibility

N = 100 # sample size
a = 0.5 # Demand intercept
b = 0.5 # Demand slope (note that we define as positive because in the demand equation there will be a negative sign attached to it)
α = 0.5 # Supply intercept
β = 0.5 # Supply slope

σ_u = 1.5 # Std of demand shocks
σ_v = 2.5 # Std of supply shocks

μ_u = 0.0 # Mean of demand shocks
μ_v = 0.0 # Mean of supply shocks

# Both demand and supply shocks are going to be mean zero and with a positive standard deviation 

# Our primary purpose is to estimate b, the structural elasticity of demand

# -------------------------------------------------------------------------- #
# Naive OLS estimation
# -------------------------------------------------------------------------- #

# Simulate observational data for supply and demand

# Create zero vectors first to later allocate values

u = zeros(N)
v = zeros(N) 

# In vectors u and v, for each i, assign a random value of the normal distribution with the corresponding mean and standard deviation. 

for i in 1:N
    u[i] = rand(Normal(μ_u, σ_u))
    v[i] = rand(Normal(μ_v, σ_v))
end

# Define a price vector, applying the expression for price in equilibrium

p = zeros(N) # First create a vector of zeros to later allocate values

# For each i, assign the value of the expression for price in equilibrium

for i in 1:N
    p[i]=((a-α)/(β+b)) + ((u[i]-v[i])/(β+b)) # Uses the random demand and supply shocks
end

# Define a quantity vector, applying the expression for quantity in equilibrium

# Vector of zeroes to later allocate values

y = zeros(N)

# For each i, assign the value of the expression for quantity in equilibrium. Also uses the random demand and supply shocks

for i in 1:N
    y[i]= ((a*β + b*α)/(b+β)) + ((β*u[i] + b*v[i])/(β+b))
end

# Demand and supply curves, with their structural expressions
# Using both the random shocks and the equilibrium prices and quantities

D = zeros(N)
S = zeros(N)

for i in 1:N
    D[i] = a - b*p[i] + u[i] 
    S[i] = α + β*p[i] + v[i]
end

# Estimate naive OLS below through the Ipopt optimizer

# Non-linear system to be solved for Pi and Pj

ols_naive = JuMP.Model(Ipopt.Optimizer)    # Initializing the optimizer

# Define variables for the optimization

JuMP.@variable(ols_naive, γ0) # Gamma 0 is the intercept

JuMP.@variable(ols_naive, γ1) # Gamma 1 is the slope

JuMP.@objective(ols_naive, Min, sum((y[i] - γ0 - γ1*p[i])^2  for i in 1:N)) # Minimize the sum of squared residuals. This is the objective function

JuMP.optimize!(ols_naive) # Perform the optimization

γ1_h = JuMP.value.(γ1)  # Access the slope. Value obtained: -0.214. Biased due to simultaneity.

# Calculate the same value according to the known formula for the slope of the OLS estimator

γ1_2 = cov(y,p)/var(p)  # Value obtained: -0.214 (in absolute value, it is 0.214, which is far from 0.5, what it should be.)

# OLS biased as the true value is 0.5

# -------------------------------------------------------------------------- #
# 2. IV estimation (2SLS) through its theoretical equation
# -------------------------------------------------------------------------- #

# Simulate data for the instruments x_u and x_v 

# Relationships of the instruments with the random demand shocks (u = c_u*x_u + ϵ_u) and supply shocks (v = c_v*x_v + ϵ_v)

c_u = 0.5
c_v = 0.5

# Standard deviations of the unobserved parts of random shocks

σ_ϵu = 0.25
σ_ϵv = 0.5

# Means of the unobserved parts of random shocks

μ_ϵ_u = 0.0
μ_ϵ_v = 0.0

# Construct the vectors for the unobserved parts of the demand shocks and supply shocks

# First create zero vectors to later allocate values

ϵ_u = zeros(N)
ϵ_v = zeros(N)

# Simulate the unobserved part as a random draw from a normal distribution with the corresponding mean and standard deviation (for each i), for supply and demand 

for i in 1:N
    ϵ_u[i] = rand(Normal(μ_ϵ_u, σ_ϵu)) 
    ϵ_v[i] = rand(Normal(μ_ϵ_v, σ_ϵv))
end 

# Now we may calculate the instruments x_u and x_v by solving for them from (u = c_u*x_u + ϵ_u) and (v = c_v*x_v + ϵ_v) - this is x_u = (u - ϵ_u)/c_u and x_v = (v - ϵ_v)/c_v

# Create zero vectors to later allocate values

x_u = zeros(N)
x_v = zeros(N)

# Use the expressions above to calculate the instruments with the loop and the coefficients c_u and c_v

for i in 1:N
    x_u[i] = (( u[i] - ϵ_u[i] ) / (c_u) )
    x_v[i] = (( v[i] - ϵ_v[i] ) / (c_v)) 
end

# Estimate b_iv with the theoretical equation for it from the notes

b1_iv2 = (cov(y,x_v)/cov(p,x_v))    # Value obtained: -0.558. This is a far better estimation than the -0.21 obtained via naive OLS.

# -------------------------------------------------------------------------- #
# 2. IV estimation (2SLS) through Ipot optimizer (assignment)
# -------------------------------------------------------------------------- #

# 2SLS takes the instrument and does two regressions: 
# 1. Regress the endogenous variable on the instrument to obtain the predicted values of the endogenous variable
# 2. Regress the predicted values of the endogenous variable on the dependent variable to obtain the coefficient estimate

# In our case, the instrument is an exogenous, observed variable which affects shocks on supply, x_v. 
# The endogenous variable is the price, p. (for the estimation of the structural demand elasticity, v)

# First stage using Ipopt

first_stage = JuMP.Model(Ipopt.Optimizer)    # Initializing the optimizer

# Define the variables for the first stage optimization problem, which is the regression of the endogenous variable on the instrument. p_hat = π0 + π1*x_v

JuMP.@variable(first_stage, π0) # Pi 0 is the intercept of the first stage regression

JuMP.@variable(first_stage, π1) # Pi 1 is the slope of the first stage regression

JuMP.@objective(first_stage, Min, sum((p[i] - π0 - π1*x_v[i])^2  for i in 1:N)) # Minimize the sum of squared residuals. This is the objective function

JuMP.optimize!(first_stage) # Perform the optimization

π0_hat = JuMP.value.(π0) # Access the intercept

π1_hat = JuMP.value.(π1) # Access the slope

p_hat = zeros(N) # Create a vector of zeros to later allocate values

# For each i, assign the value of the expression for the predicted price

for i in 1:N
    p_hat[i] = π0_hat + π1_hat*x_v[i]
end

# Second stage using Ipopt  

second_stage = JuMP.Model(Ipopt.Optimizer)    # Initializing the optimizer

# Define the variables for the second stage optimization problem, which is the regression of the predicted endogenous variable on the dependent variable. y_hat = β0 + β1*p_hat

JuMP.@variable(second_stage, γ0) # Gamma 0 is the intercept of the second stage regression

JuMP.@variable(second_stage, γ1) # Gamma 1 is the slope of the second stage regression

JuMP.@objective(second_stage, Min, sum((y[i] - γ0 - γ1*p_hat[i])^2  for i in 1:N)) # Minimize the sum of squared residuals. This is the objective function

JuMP.optimize!(second_stage) # Perform the optimization

γ1_hat = JuMP.value.(γ1) # Access the slope of the second stage regression

# Absolute value of the coefficient estimate is 0.55, the same as the one obtained in the theoretical equation for the IV estimator. 

