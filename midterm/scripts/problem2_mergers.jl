# SFU ECON832 Midterm
# Spring 2024
# Problem 2 Mergers

# Calculatations for the mergers in both scenarios

# Preliminaries -------------------------------------------------------------

using Pkg

# Project .toml

Pkg.activate(".")
Pkg.instantiate()

# Load packages

using CSV
using LinearAlgebra
using DataFrames
using TidierData
using PrettyTables

# Load raw data ------------------------------------------------------------

# Load raw simulated data for preparing the counterfactual scenario

market_shares_raw = CSV.read("data/simulated-data/midterm_simulated_market_data_s.csv", DataFrame)

attributes_raw = CSV.read("data/simulated-data/midterm_simulated_market_data_x.csv", DataFrame)

costs_raw = CSV.read("data/simulated-data/midterm_simulated_market_data_w.csv", DataFrame)

# Load demand side parameters from the baseline scenario

θ_1 = CSV.read("data/output/theta1_values.csv", DataFrame)

θ_2 = CSV.read("data/output/theta2_values.csv", DataFrame)

# Load supply side parameters from the baseline scenario

θ_3 = CSV.read("data/output/theta3_values.csv", DataFrame)

# Define the matrix of θ1

θ = Vector(θ_1[:,"theta1_values"])

# Define the matrix of θ3, but eliminate the first element (price)

θ₃ = Vector(θ_3[:, "θ3_values"])

# Load data from ξ from the baseline scenario

ξ_df = CSV.read("data/output/xi_values.csv", DataFrame)

ξ = Matrix(ξ_df)

# Data preparation -----------------------------------------------------------

# I need to prepare a scenario where MD and TH merge, and SB is left alone. 
# For caffeine scores, csMDTH = min(csMD; csTH) (both for observed consumer characteristic, and for the cost characteristic) in the efficiency scenario
# In the average scenario, csMDTH = (csMD + csTH) / 2 (both for observed consumer characteristic, and for the cost characteristic)
# There will still be three products in the market (MD = 1, TH = 2, SB = 3), but MD and TH are produced by firm MDTH, which is defined as firm 1.
# SB is produced by firm SB, which is defined as firm 2.
# Would need to recalculate the market shares using a share function as defined in the demand_functions.jl for both scenarios.

# Prices and shares are what needs to be solved fon so I do not prepare them here. 

# First for caffeine scores (consumers) in the efficiency scenario

caffeine_scores_efficiency = @chain begin
    attributes_raw
    @transmute(cdid = marketid,
               caffeine_score_MDTH_1 = min(caffeine_score_MD, caffeine_score_TH), 
               caffeine_score_MDTH_2 = min(caffeine_score_MD, caffeine_score_TH), 
               caffeine_score_SB)
    @pivot_longer(caffeine_score_MDTH_1:caffeine_score_SB, names_to = "id", values_to = "caffeine_score")
    @mutate(id = case_when(id == "caffeine_score_MDTH_1" => 1,
                           id == "caffeine_score_MDTH_2" => 2,
                           id == "caffeine_score_SB" => 3))
    @mutate(firmid = if_else(id in [1, 2], 1, 2),
            `const` = 1)
    @relocate(firmid, after = cdid)
end

# Now for the cost attributes in the efficiency scenario

cost_attributes_efficiency = @chain begin
    costs_raw
    @transmute(cdid = marketid,
               caffeine_score_MDTH_1 = min(caffeine_score_MD, caffeine_score_TH),
               caffeine_score_MDTH_2 = min(caffeine_score_MD, caffeine_score_TH),
               caffeine_score_SB)
    @pivot_longer(caffeine_score_MDTH_1:caffeine_score_SB, names_to = "id", values_to = "caffeine_score")
    @mutate(id = case_when(id == "caffeine_score_MDTH_1" => 1,
                           id == "caffeine_score_MDTH_2" => 2,
                           id == "caffeine_score_SB" => 3))
    @mutate(firmid = if_else(id in [1, 2], 1, 2),
            `const` = 1)
    @relocate(firmid, after = cdid)
end

# Now for the caffeine scores (consumers) in the average scenario

caffeine_scores_average = @chain begin
    attributes_raw
    @transmute(cdid = marketid,
               caffeine_score_MDTH_1 = (caffeine_score_MD + caffeine_score_TH) / 2, 
               caffeine_score_MDTH_2 = (caffeine_score_MD + caffeine_score_TH) / 2, 
               caffeine_score_SB)
    @pivot_longer(caffeine_score_MDTH_1:caffeine_score_SB, names_to = "id", values_to = "caffeine_score")
    @mutate(id = case_when(id == "caffeine_score_MDTH_1" => 1,
                           id == "caffeine_score_MDTH_2" => 2,
                           id == "caffeine_score_SB" => 3))
    @mutate(firmid = if_else(id in [1, 2], 1, 2),
            `const` = 1)
    @relocate(firmid, after = cdid)
end

# Now for the cost attributes in the average scenario

cost_attributes_average = @chain begin
    costs_raw
    @transmute(cdid = marketid,
               caffeine_score_MDTH_1 = (caffeine_score_MD + caffeine_score_TH) / 2,
               caffeine_score_MDTH_2 = (caffeine_score_MD + caffeine_score_TH) / 2,
               caffeine_score_SB)
    @pivot_longer(caffeine_score_MDTH_1:caffeine_score_SB, names_to = "id", values_to = "caffeine_score")
    @mutate(id = case_when(id == "caffeine_score_MDTH_1" => 1,
                           id == "caffeine_score_MDTH_2" => 2,
                           id == "caffeine_score_SB" => 3))
    @mutate(firmid = if_else(id in [1, 2], 1, 2),
            `const` = 1)
    @relocate(firmid, after = cdid)
end

# Counterfactual marginal costs for the efficiency scenario ------------------------------------------------

# Estimate the counterfactual marginal cost for all firms under ln(MC) = θ3 * caffeine_score (caffeine score is xj)

x_j_eff = Matrix(cost_attributes_efficiency[!, ["const", "caffeine_score"]]) 

ln_mc_eff = x_j_eff * θ₃

# Drop values which are negative from the ln_mc_eff vector

ln_mc_eff = ln_mc_eff[ln_mc_eff .> 0]

# Get actual marginal cost by exponentiating

MC_eff= exp.(ln_mc_eff)


# Counterfactual marginal costs for the average scenario ------------------------------------------------

# Estimate the counterfactual marginal cost for all firms under ln(MC) = θ3 * caffeine_score (caffeine score is xj)

x_j_avg = Matrix(cost_attributes_average[!, ["const", "caffeine_score"]])

ln_mc_avg = x_j_avg * θ₃

# Drop values which are negative from the ln_mc_avg vector

ln_mc_avg = ln_mc_avg[ln_mc_avg .> 0]

# Get actual marginal cost by exponentiating

MC_avg = exp.(ln_mc_avg)

