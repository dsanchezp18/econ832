# SFU ECON832 Midterm
# Spring 2024

# Libraries

library(readr)
library(dplyr)
library(here)

# Load the data

attributes_raw <- read_csv(here("data/simulated-data/midterm_simulated_market_data_x.csv"),
                           show_col_types = F)

blp_original_product_data <- read_csv("C:/Users/user/Documents/GitHub/julia-blp-modified/data and random draws/BLP_product_data.csv",
                                      show_col_types = F)

blp_original_product_data