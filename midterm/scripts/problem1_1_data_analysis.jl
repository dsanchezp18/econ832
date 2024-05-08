# SFU ECON832 Midterm
# Spring 2024
# Data Preparation Script

# Preliminaries -------------------------------------------------------------

using Pkg

# Project .toml

Pkg.activate(".")
Pkg.instantiate()

# Load packages

using CSV
using DataFrames
using TidierData
using TidierStrings
using PrettyTables

# Load data ------------------------------------------------------------------

# Load the simulated observations for market shares, prices, product attributes (caffeine score and price), cost attributes w

market_shares_raw = CSV.read("data/simulated-data/midterm_simulated_market_data_s.csv", DataFrame)

attributes_raw = CSV.read("data/simulated-data/midterm_simulated_market_data_x.csv", DataFrame)

costs_raw = CSV.read("data/simulated-data/midterm_simulated_market_data_w.csv", DataFrame)

# Data preparation -----------------------------------------------------------

## Data for demand elasticities ---------------------------------------

# I assume this data should mirror the original BLP cars dataset, with the same structure and variables.
# I have 3 products per market, and 200 markets. Thus, this data frame should have 600 rows.
# Start with the attributes data frame, pivot from wide to long for the prices and caffeine scores, later join together.

prices = @chain attributes_raw begin
    @select(marketid, price_MD, price_TH, price_SB)
    @rename(MD = price_MD, TH = price_TH, SB = price_SB)
    @pivot_longer(MD:SB, names_to = "product", values_to = "price")
end

caffeine_scores = @chain begin
    attributes_raw
    @select(marketid, caffeine_score_MD, caffeine_score_TH, caffeine_score_SB)
    @rename(MD = caffeine_score_MD, TH = caffeine_score_TH, SB = caffeine_score_SB)
    @pivot_longer(MD:SB, names_to = "product", values_to = "caffeine_score")
end

# Left join the prices and caffeine scores data frames on the marketid and product columns 
# No need to specifiy the identifiers, as it's a natural join, but I do it for clariy the first time(same column names)

attributes_long = @left_join(prices, caffeine_scores, (marketid, product))

# I now pivot the market shares data frame from wide to long, and join it with the long attributes data frame

market_shares_long = @chain begin
    market_shares_raw
    @select(marketid, MD, TH, SB)
    @pivot_longer(MD:SB, names_to = "product", values_to = "share")
    @arrange(marketid, product)
end

# To get the "outshare" variable, I need to calculate the sum of shares for each market

total_shares = @chain market_shares_long begin
    @group_by(marketid)
    @summarize(total_share = sum(share))
end

# Seemingly, the total shares per market do not total 1. 
# Left join to market_shares_long and use mutate to calculate the outshare variable

market_shares_long = @chain begin
    market_shares_long
    @left_join(total_shares, marketid)
    @mutate(outshr = total_share - share)
    @select(-total_share)
end
# Left join the market shares and attributes data frames on the marketid and product columns (natural join, so no need to specify identifiers)
# Also rename marketid column to cdid, respectively, to match the original BLP cars dataset
# For firm id, I assign 1 to MD, 2 to TH, and 3 to SB. I can't have a string as a firm id, so I need to convert the product column to a string first.
# I create the id variable (product id) and the constant, which is equal to 1 for all observationsw

product_data = @chain begin
    market_shares_long
    @left_join(attributes_long)
    @rename(cdid = marketid)
    @mutate(firmid = case_when(product == "MD" => 1, 
                               product == "TH" => 2, 
                               product == "SB" => 3))
    @select(-product)
    @relocate(firmid, cdid)
    @relocate(share, outshr, after = caffeine_score)
    @arrange(cdid, firmid)
    @mutate(id = row_number(), 
            `const` = 1)
    @relocate(id, after = cdid)
    @relocate(`const`, after = price)
end

# Add costs from the supply side -------------------------------------------------------

# Add costs to the product data frame. First need to reshape from wide to long, then join with the product data frame.

caffeine_score_costs = @chain costs_raw begin
    @pivot_longer(-marketid, names_to = "product", values_to = "caffeine_score_cost")
    @mutate(product = str_remove_all(product,"caffeine_score_"))
    @mutate(firmid = case_when(product == "MD" => 1, 
                               product == "TH" => 2, 
                               product == "SB" => 3))
    @select(-product)
    @rename(cdid = marketid)
    @arrange(cdid, firmid)
    @mutate(id = row_number())
    @relocate(firmid, cdid, id)
end

# Left join to the product data

product_data_final  = @chain product_data begin
    @left_join(caffeine_score_costs)
    @relocate(caffeine_score_cost, after = caffeine_score)
end

# Disallow missing values in the product data

product_data_final = dropmissing(product_data_final)

# Print for the screenshot

print(first(product_data_final, 60))

# Export to a .csv for later use

CSV.write("data/output/reshaped_product_data.csv", product_data_final)
