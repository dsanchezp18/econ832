# === Plots ====#

# Trying out plotting with Julia

using Plots

# Define the data

x = 1:10; y = rand(10); # These are the plotting data

# Plot the data

plot(x, y, label="my label")