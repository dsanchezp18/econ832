# SFU ECON832 Final
# Spring 2024
# Script to train the deep feedforward neural network (DFNN) with risk features only

# Preliminaries -------------------------------------------------------------------

# Activate the package environment

using Pkg

Pkg.activate("final_env")

# Parallel computing

# Load packages

using TidierData
using DataFrames
using TidierFiles
using Flux
using MLBase
using Random
using LinearAlgebra
using Parameters: @with_kw

# Load data

# Calibration data for experiment 1 (clean)

calibration_cleaned = @chain read_csv("data/output/df_exp1.csv") begin
    @clean_names()
end

# Training data (clean)

training_cleaned = @chain read_csv("data/output/df_train.csv") begin
    @clean_names()
end

# Testing data (clean)

testing_cleaned = @chain read_csv("data/output/df_test.csv") begin
    @clean_names()
end

# Competition data (clean)

competition_cleaned = @chain read_csv("data/output/cleaned_competition.csv") begin
    @clean_names()
end

# Data -------------------------------------------------------------------

## Selecting variables for the training data -------------------------------------------------------------------

# Select the relevant variables related to lotteries for the baseline model (id and subjid gets dropped for the DFNN)

df_train  = @chain training_cleaned begin
    @filter(experiment_1 == 1)
    @select(location_rehovot, gender_female, age, # Demographics
            shape_b_symm, shape_b_rskew, shape_b_lskew, # Lottery shapes B
            shape_a_symm, shape_a_rskew, shape_a_lskew, # Lottery shapes A
            ha, hb, p_ha, p_hb, # Expected values and probabilities
            lotnumb, lotnuma, lb, la, corr, amb,  # Other variables related to the lottery
            payoff, apay, bpay,
            b) # Outcome variable
end

## Selecting variables for the testing data -------------------------------------------------------------------

# Select the relevant variables related to lotteries for the baseline model (id and subjid gets dropped for the DFNN)

df_testing  = @chain testing_cleaned begin
    @filter(experiment_1 == 1)
    @select(location_rehovot, gender_female, age, # Demographics
            shape_b_symm, shape_b_rskew, shape_b_lskew, # Lottery shapes B
            shape_a_symm, shape_a_rskew, shape_a_lskew, # Lottery shapes A
            ha, hb, p_ha, p_hb, # Expected values and probabilities
            lotnumb, lotnuma, lb, la, corr, amb,  # Other variables related to the lottery
            payoff, apay, bpay,
            b) # Outcome variable
end

# DFNN -------------------------------------------------------------------

Random.seed!(593)

# Train a DFNN with risk preference variables and demographic variables
# Outcome variable is b

## Feature engineering -------------------------------------------------------------------

# Need to prepare the data for the model
# 1. Separate features from the outcome
# 2. Normalize features (mean 0, sd 1)
# 3. Transpose the matrix of features to have observations as columns

@with_kw mutable struct Args
    lr::Float64 = 0.7
    epochs::Int = 100  # Add this line
end

# Initialize hyperparameter arguments

args = Args(lr=0.7, epochs=100)

# Separate data in X and Y

features = Matrix(df_train[:, Not(:b)])

outcome = df_train.b # No need to hot encode the outcome variable since it is binary

# Standardize continuous variables, mean of 0 and sd 1

X = transpose(Flux.normalise(features, dims = 2))

# Transpose the outcome vector

Y = transpose(outcome)

# Data

data = [(X, Y)]

## Model training -------------------------------------------------------------------

# Define your model

model = Flux.Chain(
    Dense(size(X)[1], 64, Flux.relu),
    Dense(64, 64, Flux.relu),
    Dense(64, 64, Flux.relu),
    Dense(64, 64, Flux.relu),
    Dense(64, 1, Flux.sigmoid)
)

# Define loss function based on MSE

loss(X, Y) = Flux.mse(model(X), Y)

# Define optimizer: gradient descent with learning rate `args.lr`

optimiser = Descent(args.lr)

# Train model for `args.epochs` epochs

for epoch in 1:args.epochs
    Flux.train!(loss, Flux.params(model), data, optimiser)
end

# Model evaluation -------------------------------------------------------------------

# Loss

loss(X, Y)

# Predictions

predictions_train = model(X)

# Convert predictions to binary

predictions_train_binary = predictions_train .> 0.5

# Confusion matrix# Convert boolean arrays to integer arrays with class labels starting from 1
Y_int = Int.(Y .> 0.5) .+ 1
predictions_train_binary_int = Int.(predictions_train_binary) .+ 1

# Calculate confusion matrix

confusion_matrix_train = confusmat(2, vec(Y_int), vec(predictions_train_binary_int))

# Accuracy

accuracy_train = sum(diag(confusion_matrix_train)) / sum(confusion_matrix_train)

# Testing the model -------------------------------------------------------------------

# Extract X and Y from calibration_test

features_test = Matrix(df_testing[:, Not(:b)])

outcome_test = df_testing.b

X_test = transpose(Flux.normalise(features_test, dims = 2))

Y_test = transpose(outcome_test)

# Execute the model on the test data

loss(X_test, Y_test)

# Compute the confusion matrix for the test data

predictions_test = model(X_test)

predictions_test_binary = predictions_test .> 0.5

Y_test_int = Int.(Y_test .> 0.5) .+ 1

predictions_test_binary_int = Int.(predictions_test_binary) .+ 1

confusion_matrix_test = confusmat(2, vec(Y_test_int), vec(predictions_test_binary_int))

accuracy_test = sum(diag(confusion_matrix_test)) / sum(confusion_matrix_test)

# Testing the model on the competition data -------------------------------------------------------------------

# Select relevant variables from competition data

df_competition = @chain competition_cleaned begin
    @filter(experiment_1 == 1)
    @select(location_rehovot, gender_female, age, # Demographics
            shape_b_symm, shape_b_rskew, shape_b_lskew, # Lottery shapes B
            shape_a_symm, shape_a_rskew, shape_a_lskew, # Lottery shapes A
            ha, hb, p_ha, p_hb, # Expected values and probabilities
            lotnumb, lotnuma, lb, la, corr, amb,  # Other variables related to the lottery 
            payoff, apay, bpay,
            b) # Outcome variable
end

dropmissing!(df_competition)

# Extract X and Y from competition data

features_competition = Matrix(df_competition[:, Not(:b)])

outcome_competition = df_competition.b

# Feature engineering for the competition data

X_competition = transpose(Flux.normalise(features_competition, dims = 2))

Y_competition = transpose(outcome_competition)

# Execute the model on the competition data

# Loss

loss(X_competition, Y_competition)

# Compute the confusion matrix for the competition data

predictions_competition = model(X_competition)

predictions_competition_binary = predictions_competition .> 0.5

Y_competition_int = Int.(Y_competition .> 0.5) .+ 1

predictions_competition_binary_int = Int.(predictions_competition_binary) .+ 1

confusion_matrix_competition = confusmat(2, vec(Y_competition_int), vec(predictions_competition_binary_int))

accuracy_competition = sum(diag(confusion_matrix_competition)) / sum(confusion_matrix_competition)

# Export results -------------------------------------------------------------------

# Export the confusion matrix of the model on the training data

write_csv(DataFrame(confusion_matrix_train, :auto),
          "data/output/confusion_matrix_train_baseline_dfnn.csv")

# Export the confusion matrix of the model on the competition data

write_csv(DataFrame(confusion_matrix_competition, :auto),
          "data/output/confusion_matrix_competition_baseline_dfnn.csv")

# Export accuracies

accuracies_df = DataFrame(
    data = ["Train", "Test", "Competition"],
    accuracy = [accuracy_train, accuracy_test, accuracy_competition]
)

write_csv(accuracies_df, "data/output/accuracies_baseline_dfnn.csv")

# Export loss function values

losses_df = DataFrame(
    data = ["Train", "Test", "Competition"],
    loss = [loss(X, Y), loss(X_test, Y_test), loss(X_competition, Y_competition)]
)

write_csv(losses_df, "data/output/losses_baseline_dfnn.csv")