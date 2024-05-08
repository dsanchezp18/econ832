# SFU ECON832 Midterm
# Spring 2024
# Estimation of Elasticities with the provided instruments

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

# Loading my processed data from the previous question

products = CSV.read("data/output/reshaped_product_data.csv", DataFrame)

# Loading simulated instruments data

instruments_supply_raw = CSV.read("data/simulated-data/midterm_simulated_market_data_zs.csv", DataFrame)

instruments_demand_raw  = CSV.read("data/simulated-data/midterm_simulated_market_data_zd.csv", DataFrame)

# Random draws ----------------------------------------------------------------

# This time, I will estimate elasticities for two attributes: price and caffeine score.
# This means I need to do random draws for two attributes for each individual in each market.
# We have d = 2 attributes, for T = 200 markets. Can vary the sample size to test performance. 
# This means I need a 3D array of random draws of size T x s x d. 

# Define parameters (dimensions)

T = 200
s = 100 # Change to see results and evaluate runtime
d = 2
mean = 0
var = 1

# Set seed

Random.seed!(123)

# Create a multivariate standard normal distribution with variance 0.2

nu = MvNormal([mean], [var])

# Repeat for each individual in s

v = zeros(T, s, d)

for i in 1:T
    for j in 1:s
        v[i, j, :] = rand(nu, d)
    end
end

# Preparing supply instrument data -------------------------------------------------------

# I need an instrument matrix Z to do demand estimation, but the current data does not fit the required format.
# I follow a similar process to what I had before, but this time I need to do it for the supply instruments z_s which I need to estimate the elasticities.
# Pivot from wide to long format, I expect to have a 600 rows data frame. 
# Mantain the identification of MD = 1, TH = 2, and SB = 3.
# I also create my id variable, which I will need in the model.

instruments_supply_long = @chain instruments_supply_raw begin
    @relocate(marketid)
    @arrange(marketid)
    @select(marketid, startswith("zs"))
    @pivot_longer(-marketid, names_to = "zs", values_to = "value")
    @separate(zs, into = ["instrument", "product"], sep = "_")
    @mutate(firmid = case_when(product == "MD" => 1, product == "TH" => 2, product == "SB" => 3))
    @select(-product)
    @relocate(firmid, before = marketid)
    @relocate(value, after = firmid)
    @pivot_wider(names_from = "instrument", values_from = "value")
    @rename(cdid = marketid)
    @arrange(cdid, firmid)
    @mutate(id = row_number())
    @relocate(id, after = firmid)
end

# Disallow missing values in the data frame 

instruments_supply_long = dropmissing(instruments_supply_long)

# Export the reshaped instruments data to a csv file

CSV.write("data/output/reshaped_instruments_supply_data.csv", instruments_supply_long)

# Preparing demand instrument data -------------------------------------------------------

# I follow a similar process to what I had before, but this time I need to do it for the supply instruments z_d

instruments_demand_long = @chain instruments_demand_raw begin
    @relocate(marketid)
    @arrange(marketid)
    @select(marketid, startswith("zd"))
    @pivot_longer(-marketid, names_to = "zd", values_to = "value")
    @separate(zd, into = ["instrument", "product"], sep = "_")
    @mutate(firmid = case_when(product == "MD" => 1, product == "TH" => 2, product == "SB" => 3))
    @select(-product)
    @relocate(firmid, before = marketid)
    @relocate(value, after = firmid)
    @pivot_wider(names_from = "instrument", values_from = "value")
    @rename(cdid = marketid)
    @arrange(cdid, firmid)
    @mutate(id = row_number())
    @relocate(id, after = firmid)
end

# Disallow missing values in the data frame

instruments_demand_long = dropmissing(instruments_demand_long)

# Export the reshaped instruments data to a csv file

CSV.write("data/output/reshaped_instruments_demand_data.csv", instruments_demand_long)

# Since I reordered the df in the correct order, I assume that I will be able to use this data frame in the matrices without any problems.

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

# Define the instruments matrix Z, without ids, which has only the instruments I need for the demand estimation

instruments_matrix = @chain instruments_demand_long begin
    @select(startswith("zd"))
end

Z = Matrix(instruments_matrix)

# Demand elasticities with Optim -------------------------------------------------------

# I load the functions, copied from the JuliaBLP repository, that I need to estimate the demand parameters.
# These are modified versions of the functions in the repository, which I adapted to the dimensions of the simulated data
# I also implement best practices as per the PyBLP article (1E-14 and 1E-12 tolerance.)

include("functions/demand_functions.jl")    # module with custom BLP functions (objective function and σ())
include("functions/demand_derivatives.jl")   # module with gradient function

using .demand_functions
using .demand_derivatives

# I minimize the objective function to get the demand elasticities.
# Define the initial values for the parameters

θ₂ = [0.0, 0.0] # this implies starting θ₁ values equal to the IV coefficients (random effects = 0)

# Test run

# Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid)
# g = gradient(θ₂,X,Z,v_50,cdid,ξ,𝒯)
# Works normally, can use within functions

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

#result = optimize(f, θ₂, NelderMead(), Optim.Options(x_tol=1e-3, iterations=500, show_trace=true, show_every=10)) Not recommended by the PyBLP article
result = optimize(f, ∇, θ₂, LBFGS(), Optim.Options(x_tol=1e-12, iterations=50, show_trace=true, show_every=1)) 
result = optimize(f, ∇, θ₂, BFGS(), Optim.Options(x_tol=1e-12, iterations=50, show_trace=true, show_every=1))
result = optimize(f, ∇, θ₂, GradientDescent(), Optim.Options(x_tol=1e-12, iterations=50, show_trace=true, show_every=1))

# get results 
θ₂ = Optim.minimizer(result)
θ₁ = demand_objective_function(θ₂,X,share,Z,v,cdid)[2]
ξ = demand_objective_function(θ₂,X,share,Z,v,cdid)[3]

# I specified price, constant and caffeine score in the matrix.
# θ₁ = [-1.05 , -1.06 . 2.05]
# First element is the price elasticity, second is the constant, and third is the caffeine score elasticity.

# Supply elasticity -------------------------------------------------------

# I now estimate the supply elasticities using 2SLS (and compare with OLS)

# Do an X matrix for the supply side, with all observables for the supply side except price

X_s = Matrix(products[!, ["const", "caffeine_score_cost"]])

# Do the same, but with price

X_s2 = Matrix(products[!, ["const", "price", "caffeine_score_cost"]])

# Vector of prices

P = Vector(products[!, "price"])

# Observed market shares

S = Vector(products[!, "share"])

# Firm and market identifiers

firm_id = Vector(products[!, "firmid"])

market_id = Vector(products[!, "cdid"])

# Use my estimates for θ₁ and θ₂ to estimate the supply elasticities

theta_1s = θ₁

theta_2s = θ₂

# Define a random draw for a higher number of individuals individuals in each of the 200 markets

v_more = zeros(T, 5000, d)

for i in 1:T
    for j in 1:5000
        v_more[i, j, :] = rand(nu, d)
    end
end

# Define the instruments matrix, which does not contain any ids

instruments_supply_matrix = @chain instruments_supply_long begin
    @select(startswith("zs"))
end

Z_s_prelim= Matrix(instruments_supply_matrix)

# Notice that since both instruments and products are ordered by market and firm, I can use the same index to drop the same rows in both matrices.

## Non-competitive pricing -------------------------------------------------------

# Do not assume P = MC

# Import supply price elasticity functions, but having fixed the dimensions issue with the module

include("functions/supply_price_elasticities.jl")
using .supply_price_elasticities

# Matrix of price elasticities

Δ = price_elasticities(theta_1s, theta_2s, X_s2, S, v_more, v, market_id, firm_id)

# Get the inverse

Δ_inv = inv(Δ)

# Calculate marginal cost

MC = P - Δ_inv*S

# drop any negative marginal cost estimates to allow for log transformation

positive_MC_index = MC.>0

MC_pos = MC[positive_MC_index]

X_sp = X_s[positive_MC_index,:]

# Drop the same amount of rows in the instruments matrix (they are also ordered)

Z_s = Z_s_prelim[positive_MC_index,:]

# Parameter estimates using OLS

θ_3_ols = inv(X_sp'X_sp)X_sp'log.(MC_pos)

# θ3 is 0.23 (the cost elasticity with OLS might be biased)

# Repeat using 2SLS, using the instruments given by the caffeine score cost attribute

# Regress the endogenous variable (cost characteristic) on the instruments to get the first stage parameter of the cost characteristic

π_hat = inv(Z_s'Z_s)Z_s'X_sp

# Calculate the predicted values of the cost characteristic

X_hat = (Z_s)π_hat

# Regress MC on the predicted values of the cost characteristic

θ_3_2sls = inv(X_hat'X_hat)X_hat'log.(MC_pos)

# I specified constant and a caffeine score cost in the matrix.
# θ3 is the second element of the cost, 0.2281 is the cost elasticity.

# Export elasticities -------------------------------------------------------

# Recover all of the elasticities from parameter vectors for a nice table

# Demand elasticities

price_elasticity = θ₁[1]
caffeine_score_elasticity = θ₁[3]

# Average cost elasticity of caffeine score

cost_elasticity = θ_3_2sls[2]

# Export the full θ3 to a csv file

θ3_df = DataFrame(θ3_values = θ_3_2sls)

CSV.write("data/output/theta3_values.csv", θ3_df)

# Create a data frame for exporting the results of the elasticities (for my report)

elasticities = DataFrame(
    attribute = ["Price", "Caffeine Score", "Caffeine Cost"],
    elasticity = [price_elasticity, caffeine_score_elasticity, cost_elasticity])
    
# Export the results to a csv file

CSV.write("data/output/elasticities_1_2.csv", elasticities)

# Export the results of θ₁ to a csv file

theta1_df = DataFrame(theta1_values = θ₁)

CSV.write("data/output/theta1_values.csv", theta1_df)

# Export the results of θ₂ to a csv file

theta2_df = DataFrame(theta_values = θ₂)

CSV.write("data/output/theta2_values.csv", theta2_df)

# Export ξ to a csv file

ξ_df = DataFrame(ξ_values = ξ)

CSV.write("data/output/xi_values.csv", ξ_df)