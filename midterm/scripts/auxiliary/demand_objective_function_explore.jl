# SFU ECON832 Midterm
# Spring 2024
# Estimation of Elasticites

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
# I choose s = 50 samples (50 individuals per market) and d = 2 attributes, for T = 200 markets.
# This means I need a 3D array of random draws of size T x s x d. (200x50x2)

# Define parameters (dimensions)

T = 200
s = 50
d = 2
var = 0.2

# Set seed

Random.seed!(98426)

# Create a multivariate standard normal distribution with variance 0.2

nu = MvNormal([0], [0.2])


# Repeat for the 50 individuals in each of the 20 markets

v_50 = zeros(T, s, d)

for i in 1:T
    for j in 1:s
        v_50[i, j, :] = rand(nu, d)
    end
end

# Preparing supply instrument data -------------------------------------------------------

# I need an instrument matrix Z to do demand estimation, but the current data does not fit the required format.
# I follow a similar process to what I had before, but this time I need to do it for the supply shocks z_s which I need to estimate the elasticities.
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

 # The loaded instruments data already has the explanatory variables included in the data frame, so I can use it directly.

 # Disallow missing values in the data frame

instruments_supply_long = dropmissing(instruments_supply_long)

# Defining matrices for the model -------------------------------------------------------

# I define the matrix and vector objects that I need for modeling.

# Define X (only price and caffiene score), as a matrix object

X = Matrix(products[!, ["price", "const", "caffeine_score"]])

# Define y, market share, as a vector object

share = Vector(products[!, "share"])

# Get identifiers for products (ids), markets (cdid), and firms (firmid)

id = Vector(products[!, "id"])

cdid = Vector(products[!, "cdid"])

firmid = Vector(products[!, "firmid"])

# Define the instruments matrix Z, which has both of my attributes (price and caffeine score) and the instruments, nothing else. 

Z = Matrix(instruments_supply_long)

# Explore the demand objective function -------------------------------------------------------

δ = zeros(size(share))

n_individuals = size(v_50, 2)

n_products = size(share, 1)

δ = repeat(δ, 1, n_individuals)

# calculate mu for each product and individual

θ₂ = [0.0, 0.0]

μ = zeros(n_products, n_individuals)

for market in unique(cdid)
    μ[cdid.==market,:] = X[cdid.==market,Not(3)] * (v_50[market,:,:] .* θ₂')' 
end

∑ₖexp = zeros(size(μ))
# for each market
for market in unique(cdid)
    # get the sequence of denominator terms for each individual
    denom_sequence = exp.(δ[cdid.==market,:] + μ[cdid.==market,:])
    # sum over all products in market for each individual
    market_denominator = sum(denom_sequence, dims=1)
    # assign to each row for given individual in given market
    ∑ₖexp[cdid.==market,:] = repeat(market_denominator, sum(cdid.==market))
end

# calculate market share for each product for each individual (2217 products x 50 individuals)
𝒯 = exp.(δ+μ) ./ (1 .+ ∑ₖexp)
# average across individuals (Monty Carlo integration)
σ = mean(𝒯, dims=2)[:]   # semicolon to make 2217x1 and get rid of hidden second dimension  

# calculate the objective function

include("../demand_functions.jl")

using .demand_functions

Q, θ₁, ξ, 𝒯 = demand_objective_function(θ₂,X,share,Z,v_50,cdid) 

# Explore demand gradient -------------------------------------------------------

n_products = size(X,1)
n_individuals = size(v_50,2)
n_coefficients = size(θ₂,1)

W = inv(Z'Z)
∂Q_∂ξ = 2*(Z'ξ)'W*Z'

∂σᵢ_∂δ = zeros(n_products, n_products, n_individuals)

diagonal_index = CartesianIndex.(1:n_products, 1:n_products) # object of (1,1) (2,2) ... (2217,2217) indices

    # calculate the derivative given 𝒯(j,i) values from objective function
    for individual in 1:n_individuals

        # derivative for off-diagonal elements: -𝒯ⱼᵢ * 𝒯ₘᵢ
        ∂σᵢ_∂δ[:,:,individual] = -𝒯[:,individual] * 𝒯[:,individual]'

        # derivative for diagonal elements: 𝒯ⱼᵢ * (1 - 𝒯ⱼᵢ)
        ∂σᵢ_∂δ[diagonal_index, individual] = 𝒯[:,individual] .* (1 .- 𝒯[:,individual])

    end

    
    # calculate mean over all individuals (Monty Carlo integration)
    ∂σ_∂δ = mean(∂σᵢ_∂δ, dims=3)[:,:] # semicolon to remove the dropped third dimension

    # calculate inverse 
    ∂σ_∂δ⁻¹ = zeros(size(∂σ_∂δ))
    # must be done market-by-market: products outside of given market do not affect shares within market (creates a block matrix)
    for market in unique(cdid)
        ∂σ_∂δ⁻¹[cdid.==market, cdid.==market] = inv(∂σ_∂δ[cdid.==market, cdid.==market])
    end

    ∂σᵢ_∂θ₂ = zeros(n_products, n_individuals, n_coefficients)

    # calculate market-by-market
    for market in unique(cdid)
        # for each of the 5 coefficients
        for coef in 1:n_coefficients

            # calculate sum term for simplicity: ∑ₖ x₁ⱼₜ 𝒯ⱼ
            Σⱼx₁ⱼ𝒯ⱼᵢ = X[cdid.==market, coef]' * 𝒯[cdid.==market,:]

            # calculate derivative for all individuals for given coefficient: v₁ᵢ 𝒯ⱼ (x₁ⱼₜ - ∑ₖ x₁ⱼₜ 𝒯ⱼ)
            ∂σᵢ_∂θ₂[cdid.==market,:,coef] = v_50[market,:,coef]' .* 𝒯[cdid.==market,:] .* (X[cdid.==market,coef] .- Σⱼx₁ⱼ𝒯ⱼᵢ)
        end
    end

        # dimension 2 indexes the 50 individuals
        ∂σ_∂θ₂ = mean(∂σᵢ_∂θ₂, dims=2)[:,1,:] # semicolons/slicing to removed the dropped dimension



        # 3. Combine derivaties to calculate gradient

# The full gradient is: 
# Q/∂θ₂ ∂= ∂Q/∂ξ * ∂ξ/∂θ₂ = (2[Z'ξ]'W*Z') (I-M)*-[∂σ/∂δ]⁻¹[∂σ/∂θ₂] ≈ 2[Z'ξ]'W*Z'(-[∂σ/∂δ]⁻¹)[∂σ/∂θ₂] 

# gradient calculation
# ∂Q_∂θ₂ = (2*(Z'ξ)'W) * Z' * -∂σ_∂δ⁻¹ * ∂σ_∂θ₂
∂Q_∂θ₂ = ∂Q_∂ξ * (-∂σ_∂δ⁻¹ * ∂σ_∂θ₂)



# run time = 1.24 s
# @btime gradient($θ₂,$X,$Z,$v_50,$cdid,$ξ,$𝒯)