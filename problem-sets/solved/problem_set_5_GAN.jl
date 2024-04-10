## Problem Set 5: Generative Adversarial Networks (GANs) in Julia
## ECON832 Computational Methods in Economics

# Preliminaries ---------------------------------------------------------------

using Pkg

# Create and activate a new environment
Pkg.activate("scripts/GAN_env")
# Pkg.add("Flux")
# Pkg.add("Distributions")
# Pkg.add("Plots")
#Pkg.add("Zygote")
#Pkg.add("Plots")

using Random
using Statistics
using Flux
using Flux: params
using Flux.Optimise: update!
using Zygote: gradient
using Plots

## Set a fixed random seed for reproducibility

Random.seed!(593)

# Data Generating Process (DGP) ---------------------------------------------------------------

# Hyperparameters

epochs = 20000
batch_size = 32
latent_size = 5
lr = 0.001

# Create a function which generates data from a normal distribution with batch_size, mu and sigma as parameters
# The function should only return samples.

function generate_data(batch_size, mu, sigma)
    return randn(Float32, 1, batch_size) .* sigma .+ mu
end

# Function to create the random noise

function sample_noise(batch_size, latent_size)
    return randn(Float32, batch_size, latent_size)
end

# Training the GAN ---------------------------------------------------------------

# Define the Generator and Discriminator networks using Flux

Generator() = Chain(Dense(latent_size + 2, 64, relu), BatchNorm(64, relu), Dense(64, 1))

Discriminator() = Chain(Dense(3, 64, relu), BatchNorm(64, relu),Dense(64, 1, σ))


G = Generator()
D = Discriminator()

# Optimizers
opt_G = ADAM(lr)
opt_D = ADAM(lr)

# Enter the real data parameters that you want to use for training

mu_given = 0
sigma_given = 1

# Training loop
for epoch in 1:epochs
    # Sample real data
    real_samples = generate_data(batch_size, mu_given, sigma_given)
    means = fill(mu_given, 1, batch_size)
    stds = fill(sigma_given, 1, batch_size)
    real_conditions = vcat(means, stds)

    # Sample noise and generate fake data
    noise = sample_noise(batch_size, latent_size)
    gen_input = vcat(noise', real_conditions)
    fake_samples = G(gen_input)

    # Train the Discriminator
    d_loss() = -mean(log.(D(vcat(real_samples, real_conditions)))) - mean(log.(1 .- D(vcat(fake_samples, real_conditions))))
    grads_D = gradient(() -> d_loss(), params(D))
    update!(opt_D, params(D), grads_D)

    # Train the Generator
    noise = sample_noise(batch_size, latent_size)
    gen_input = vcat(noise', real_conditions)
    g_loss() = -mean(log.(D(vcat(G(gen_input), real_conditions))))
    grads_G = gradient(() -> g_loss(), params(G))
    update!(opt_G, params(G), grads_G)

    # Print losses
    if epoch % 500 == 0
        println("Epoch: $epoch | Discriminator Loss: $(d_loss()) | Generator Loss: $(g_loss())")
    end
end

# Test the Generator ---------------------------------------------------------------

μ=10
σst=10
size=10000
test_noise = sample_noise(size, latent_size)
test_mean = fill(Float32(μ*1.0), size)
test_std = fill(Float32(σst*1.0), size)
test_input = vcat(test_noise', test_mean', test_std')
generated_samples = G(test_input)
mean(generated_samples)
std(generated_samples)
histogram(vec(generated_samples), bins=30, xlabel="Value", ylabel="Frequency", label="Generated samples", title="Generated Samples Distribution", legend=:topright)
