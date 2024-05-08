# Parallel Computing Example
# SFU ECON832 Computational Methods for Economics
# Spring 2024

using Distributed

# Add 4 workers
addprocs(4)

# Ensure Distributed is available on all workers
@everywhere using Distributed

# Define the task: calculate the sum of squares of the first N natural numbers
N = 10000  # You can adjust N to see different performances

# Parallel computation using @distributed macro
# We also use a reduction operation (+) to combine the results
sum_of_squares = @distributed (+) for i=1:N
    i^2
end

# Output the result
println("The sum of squares of the first $N natural numbers is: $sum_of_squares")
