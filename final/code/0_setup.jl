# SFU ECON832 Final
# Spring 2024
# Julia Project Setup Script

# Environment creation -------------------------------------------------------------

using Pkg

# Create and activate a new environment
Pkg.generate("final_env")
Pkg.activate("final_env")

# Install packages -----------------------------------------------------------------

# Install packages that will be needed for the project

Pkg.add("DataFrames")
Pkg.add("CSV")
Pkg.add("TidierData")
Pkg.add("TidierFiles")
Pkg.add("Flux")
Pkg.add("Random")
Pkg.add("MLBase")
Pkg.add("LinearAlgebra")
Pkg.add("Parameters")
Pkg.add("MLJ")
Pkg.add("IJulia")
Pkg.add("TidierPlots")
Pkg.add("HypothesisTests")
Pkg.add("PrettyTables")
Pkg.add("Plots")

# Both TidierData and Flux load Chain, so must always call Flux.Chain