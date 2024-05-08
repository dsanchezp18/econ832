# SFU ECON832 Midterm
# Spring 2024
# Estimation of Elasticities with the other instruments

# Preliminaries -------------------------------------------------------------

using Pkg

# Project .toml

Pkg.activate(".")
Pkg.instantiate()

# Load packages

using CSV               # loading data
using DataFrames        # loading data
using LinearAlgebra     # basic math
using Distributions    # for multivariate gaussian
using Statistics        # for mean
using Random            # for random seed
using TidierData       # for data manipulation
using Optim             # for optimization
using BenchmarkTools    # for benchmarking
using JuMP              # for optimization (better than Optim)
using Ipopt             # for optimization (better than Optim)

# Loading my processed data from the previous question

products = CSV.read("data/output/reshaped_product_data.csv", DataFrame)

# Defining matrices for the model -------------------------------------------------------

# I define the matrix and vector objects that I need for modeling.

# Define X (only price and caffeine score), as a matrix object

X = Matrix(products[!, ["price", "const", "caffeine_score"]])

# Define y, market share, as a vector object

share = Vector(products[!, "share"])

# Get identifiers for products (ids), markets (cdid), and firms (firmid)

id = Vector(products[!, "id"])

cdid = Vector(products[!, "cdid"])

firmid = Vector(products[!, "firmid"])

# Define the BLP instruments function

#= Two sets of instruments
1. Characteristics of other products from the same company in the same market.
Logic: the characteristics of other products affect the price of a 
given product but not its demand. Alternatively, firms decide product characteristics X 
before observing demand shocks ξ. 
2. Characteristics of other products from different companies in the same market.
Logic: the characteristics of competing products affects the price of a
given product but not its demand. Alternatively, other firms decide their product
characteristics X without observing the demand shock for the given product ξ.
=#

function BLP_instruments(X, id, cdid, firmid)
    n_products = size(id,1) # number of observations = 600
    # initialize arrays to hold the two sets of 3 instruments. 
    IV_others = zeros(n_products,1)
    IV_rivals = zeros(n_products,1)
    # loop through every product in every market (every observation)
    for j in 1:n_products
        # 1. Set of instruments from other product characteristics
        # get the index of all different products (id) made by the same firm (firmid)
        # in the same market/year (cdid) 
        #other_index = (firmid.==firmid[j]) .* (cdid.==cdid[j]) .* (id.!=id[j])
        # x variable values for other products (excluding price)
        other_x_values = #X[other_index,:]
        # sum along columns
        #IV_others[j,:] = sum(other_x_values, dims=1)
        # 2. Set of instruments from rival product characteristics
        # get index of all products from different firms (firmid) in the same market/year (cdid)
        rival_index = (firmid.!=firmid[j]) .* (cdid.==cdid[j])
        # x variable values for other products (excluding price)
        rival_x_values = X[rival_index,:]
        # sum along columns
        IV_rivals[j,:] = sum(rival_x_values, dims=1)
    end
    # vector of observations and instruments
    IV = [X IV_rivals]
    return IV
end

# Apply the instruments function to the data (X without the price), and store the result in a variable

X_for_IV = Matrix(products[!, ["caffeine_score"]])

Z = BLP_instruments(X_for_IV, id, cdid, firmid)

# Add the constant to the Z matrix (didn't go in the function, unsure why)

Z = [ones(size(Z,1)) Z]

# Random draws ----------------------------------------------------------------

# Define parameters (dimensions)

T = 200
s = 800 
d = 2
mean = 0
var = 1

# Set seed

Random.seed!(123)

# Create a multivariate standard normal distribution with variance 0.2

nu = MvNormal([mean], [var])

# Repeat for the 50 individuals in each of the 20 markets

v = zeros(T, s, d)

for i in 1:T
    for j in 1:s
        v[i, j, :] = rand(nu, d)
    end
end

# Demand elasticities with Optim ------------------------------------------------

# Load functions

include("functions/demand_functions.jl")    # module with custom BLP functions (objective function and σ())
include("functions/demand_derivatives.jl")   # module with gradient function

using .demand_functions
using .demand_derivatives

θ₂ = [0.0, 0.0] # this implies starting θ₁ values equal to the IV coefficients (random effects = 0)

# Test run
#Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v,cdid)
# g = gradient(θ₂,X,Z,v_50,cdid,ξ,𝒯)
# Worked normally, can use within functions

# Test run with the other instruments
# Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z_alt,v,cdid)
# Worked, can use within functions

function f(θ₂)
    # run objective function and get key outputs
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v,cdid)
    # return objective function value
    return Q
end

function ∇(storage, θ₂)
    # run objective function to update ξ and 𝒯 values for new θ₂
    Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v,cdid)
    # calculate gradient and record value
    g = gradient(θ₂,X,Z,v,cdid,ξ,𝒯)
    storage[1] = g[1]
    storage[2] = g[2]
end

#result = optimize(f, θ₂, NelderMead(), Optim.Options(x_tol=1e-3, iterations=500, show_trace=true, show_every=10)) # Not recommended by the PyBLP article
result = optimize(f, ∇, θ₂, LBFGS(), Optim.Options(x_tol=1e-12, iterations=50, show_trace=true, show_every=1))
result = optimize(f, ∇, θ₂, BFGS(), Optim.Options(x_tol=1e-12, iterations=50, show_trace=true, show_every=1))
result = optimize(f, ∇, θ₂, GradientDescent(), Optim.Options(x_tol=1e-12, iterations=50, show_trace=true, show_every=1))

# get results 
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v,cdid)[2]

# θ₁ = [-1.72 , 0.2187. 2.129] 
# This means that demand elasticity is -1.72, the constant is 0.2187, and the caffeine elasticity is 2.129 (positive).

# Supply elasticity -----------------------------------------------------------

# I will use the same parameters as in my other script

X_s= Matrix(products[!, ["const", "caffeine_score_cost"]])

X_s2 = Matrix(products[!, ["price","const", "caffeine_score_cost"]])

P = Vector(products[!, "price"])

S = Vector(products[!, "share"])

firm_id = Vector(products[!, "firmid"])

market_id = Vector(products[!, "cdid"])

theta_1s = θ₁

theta_2s = θ₂

v_more = zeros(T, 5000, d)

for i in 1:T
    for j in 1:5000
        v_more[i, j, :] = rand(nu, d)
    end
end

## Non-competitive pricing -------------------------------------------------------

include("functions/supply_price_elasticities.jl")
using .supply_price_elasticities

# Matrix of price elasticities

Δ = price_elasticities(theta_1s, theta_2s, X_s2, S, v_more, v, market_id, firm_id)

Δ_inv = inv(Δ)

MC = P - Δ_inv *S

positive_MC_index = MC.>0

MC_pos = MC[positive_MC_index]

X_sp = X_s[positive_MC_index,:]

id_new = id[positive_MC_index]

cdid_new = cdid[positive_MC_index]

firmid_new = firmid[positive_MC_index]

# With OLS first (no instruments, regress log(MC) on cost characteristics and an intercept

θ_3_ols = inv(X_sp'X_sp)X_sp'log.(MC_pos)

# OLS estimate of θ_3 = [-0.57, 1.05], potentially unbiased if instruments are valid

#  Calculate instruments for the supply side

n_products = size(id_new,1)
Z_s = zeros(n_products,2)

for j in 1:n_products
    rival_index = (firmid_new.!=firmid_new[j]) .* (cdid_new.==cdid_new[j])
    rival_x_values = X_sp[rival_index,:]
    Z_s[j,:] = sum(rival_x_values, dims=1)
end

# Calculate π_hat

π_hat = inv(Z_s'Z_s)Z_s'X_sp

# Perform 2SLS with this instruments, again, rivals only

X_hat = (Z_s)π_hat

θ_3_2sls = inv(X_hat'X_hat)X_hat'log.(MC_pos)

# Export elasticities --------------------------------------------------------

# Recover all of the elasticities from parameter vectors for a nice table

# Demand elasticities

price_elasticity = θ₁[1]

caffeine_score_elasticity = θ₁[3]

# Supply elasticities

cost_elasticity_ols = θ_3_ols[2]

cost_elasticity_2sls = θ_3_2sls[2]

# Create a data frame for exporting the results

elasticities = DataFrame(
    attribute = ["Price", "Caffeine Score", "Caffeine Cost (OLS)", "Caffeine Cost (2SLS)"],
    elasticity = [price_elasticity, caffeine_score_elasticity, cost_elasticity_ols, cost_elasticity_2sls]
)

# Export the results to a csv file

CSV.write("data/output/elasticities_1_3.csv", elasticities)