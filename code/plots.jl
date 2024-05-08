# === Plots ====#

# Trying out plotting with Julia

using Plots

# Define the data

x = 1:10; y = rand(10); # These are the plotting data

# Plot the data

plot(x, y, label="my label")

# For TidierPlots (ggplot2 implementation in Julia)

using TidierPlots
using DataFrames
using PalmerPenguins

penguins = dropmissing(DataFrame(PalmerPenguins.load()))

ggplot(data = penguins) + 
    geom_bar(@aes(x = species)) +
    labs(x = "Species")
