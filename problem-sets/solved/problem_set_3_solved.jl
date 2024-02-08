# Problem Set 3
# SFU ECON832 Computational Methods for Economics
# Spring 2024

# Problem 2 ---------------------------------------------------------------------

# Use bootstrapping to provide boostrap standard errors for β/b in the supply and demand model. 

### Preliminaries --------------------------------------------------------------

# Import packages

import JuMP 
import Ipopt 
using Random 
using Distributions 

# Predefine parameters

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
