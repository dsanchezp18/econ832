using DataFrames        # for Not() and sample data
using LinearAlgebra     # basic math
using Statistics        # for mean()
using CSV               # loading data

products = CSV.read("data/output/reshaped_product_data.csv", DataFrame)

T = 200
s = 100 # Change to see results and evaluate runtime
d = 2
mean = 0
var = 1

# Load data from previous estimation

theta_1_df = CSV.read("data/output/theta1_values.csv", DataFrame)

theta_2_df = CSV.read("data/output/theta2_values.csv", DataFrame)

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

X = Matrix(products[!, ["price", "const", "caffeine_score"]])

# Define y, market share, as a vector object

share = Vector(products[!, "share"])

# Get identifiers for products (ids), markets (cdid), and firms (firmid)

id = Vector(products[!, "id"])

cdid = Vector(products[!, "cdid"])

firmid = Vector(products[!, "firmid"])

# Calculating the sigma σ (predicted shares)

# Needs a delta, use the one which I got from the estimation

