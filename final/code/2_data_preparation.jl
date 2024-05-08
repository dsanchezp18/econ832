# SFU ECON832 Final
# Spring 2024
# Data preparation for deep feedforward neural networks (DFNNs)

# Preliminaries -------------------------------------------------------------------

# Activate the package environment

using Pkg

Pkg.activate("final_env")

# Load packages

using TidierData
using DataFrames
using TidierFiles
using Random

# Load data

# Calibration data

calibration_raw = @chain begin
        read_csv("data/calibration/All estimation raw data.csv")
        @clean_names()
    end

@glimpse(calibration_raw)

# Competition data

competition_raw = @chain read_csv("data/competition/individual-track/raw-comp-set-data-Track-2.csv") begin
        @clean_names()
    end

# Data preparation -------------------------------------------------------------------

## Calibration (training) data preparation -------------------------------------------------------------------

# Prepare data for DFNNs
# Categorical variables should be made into binary variables

cleaned_calibration_df = @chain calibration_raw begin
    @mutate(id = row_number(),
            subjid = as_string(subjid),
            location_rehovot = if_else(location == "Rehovot", 1, 0),
            gender_female = if_else(gender == "F", 1, 0),
            set_1 = if_else(set == 1, 1, 0),
            set_2 = if_else(set == 2, 1, 0),
            set_3 = if_else(set == 3, 1, 0),
            set_4 = if_else(set == 4, 1, 0),
            set_5 = if_else(set == 5, 1, 0),
            set_6 = if_else(set == 6, 1, 0),
            experiment_1 = if_else(set in (5,6), 1, 0), #CPC18 experiment using sets 5 and 6
            button_r = if_else(button == "R", 1, 0), # Potential attention check
            shape_a_symm = if_else(lotshapea == "Symm", 1, 0), # Lot shape A Symm
            shape_b_symm = if_else(lotshapeb == "Symm", 1, 0), # Lot shape B Symm
            shape_a_rskew = if_else(lotshapea == "R-skew", 1, 0), # Lot shape A R-skew
            shape_b_rskew = if_else(lotshapeb == "R-skew", 1, 0), # Lot shape B R-skew
            shape_a_lskew = if_else(lotshapea == "L-skew", 1, 0), # Lot shape A L-skew
            shape_b_lskew = if_else(lotshapeb == "L-skew", 1, 0), # Lot shape B L-skew
            rt = as_integer(missing_if(rt, "NA")) # convert to missing values
    )
end

## Competition (test) data preparation -------------------------------------------------------------------

# Prepare data for DFNN testing

cleaned_competition_df = @chain competition_raw begin
    @mutate(id = row_number(),
            subjid = as_string(subjid),
            location_rehovot = if_else(location == "Rehovot", 1, 0),
            gender_female = if_else(gender == "F", 1, 0),
            set_1 = if_else(set == 1, 1, 0),
            set_2 = if_else(set == 2, 1, 0),
            set_3 = if_else(set == 3, 1, 0),
            set_4 = if_else(set == 4, 1, 0),
            set_5 = if_else(set == 5, 1, 0),
            set_6 = if_else(set == 6, 1, 0),
            experiment_1 = if_else(set in (5,6), 1, 0), #CPC18 experiment using sets 5 and 6
            button_r = if_else(button == "R", 1, 0), # Potential attention check
            shape_a_symm = if_else(lotshapea == "Symm", 1, 0), # Lot shape A Symm
            shape_b_symm = if_else(lotshapeb == "Symm", 1, 0), # Lot shape B Symm
            shape_a_rskew = if_else(lotshapea == "R-skew", 1, 0), # Lot shape A R-skew
            shape_b_rskew = if_else(lotshapeb == "R-skew", 1, 0), # Lot shape B R-skew
            shape_a_lskew = if_else(lotshapea == "L-skew", 1, 0), # Lot shape A L-skew
            shape_b_lskew = if_else(lotshapeb == "L-skew", 1, 0), # Lot shape B L-skew
            rt = as_integer(missing_if(rt, "NA")) # Convert reaction time to float
    )
end

# Test-train split -------------------------------------------------------------------

# Perform a test-train split of the calibration data, but only focusing on Experiment 1

# Filter for Experiment 1

df_exp1 = @chain cleaned_calibration_df begin
    @filter(experiment_1 == 1)
end

dropmissing(df_exp1)

# Test-train split of 80-20

Random.seed!(593)

# Use slice_sample with prop = 0.8 to produce the training data

df_train = @chain df_exp1 begin
    @slice_sample(prop = 0.8, replace = false)
end

dropmissing(df_train)

# Use an antijoin to produce the testing data

df_test = @chain df_exp1 begin
    @anti_join(df_train, id)
end

dropmissing(df_test)

# Verify proportions

nrow(df_train)/nrow(df_exp1)

nrow(df_test)/nrow(df_exp1)

# Exporting -------------------------------------------------------------------

# Export the cleaned calibration data

write_csv(cleaned_calibration_df, "data/output/cleaned_calibration.csv")

# Export experiment 1 data

write_csv(df_exp1, "data/output/df_exp1.csv")

# Export training and testing data

write_csv(df_train, "data/output/df_train.csv")

write_csv(df_test, "data/output/df_test.csv")

# Export the cleaned competition data

write_csv(cleaned_competition_df, "data/output/cleaned_competition.csv")