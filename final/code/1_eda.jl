# SFU ECON832 Final
# Spring 2024
# Julia Project Setup Script

# Preliminaries -------------------------------------------------------------------

# Activate the package environment

using Pkg

Pkg.activate("final_env")

# Load packages

using TidierData
using DataFrames
using TidierFiles

# Load data 

# Calibration data

calibration_raw = @chain read_csv("data/calibration/All estimation raw data.csv") begin
        @clean_names()
        @arrange(set)
    end

# Competititon data (for testing)

competition_raw_game_block = @chain read_csv("data/competition/individual-track/Data-to-predict-Track-2.csv") begin
        @clean_names()
    end

@glimpse(competition_raw_game_block)

competition_raw = @chain read_csv("data/competition/individual-track/raw-comp-set-data-Track-2.csv") begin
        @clean_names()
end

@glimpse(competition_raw)

# Exploratory data analysis (EDA) -------------------------------------------------------

# Glimpse

@glimpse(calibration_raw)

# How many observations (rows) per subjectid 

obs_per_subject = @chain calibration_raw begin
    @group_by(subjid)
    @summarize(n_obs = n())
    @arrange(n_obs)
end

# Count how many sets per subjid (should be just 1)

sets_per_subject = @chain calibration_raw begin
    @group_by(subjid, set)
    @summarize(n_sets = n())
    @ungroup()
    @select(-n_sets)
    @group_by(subjid)
    @summarize(n_sets = n())
end

# Set 1 data -----------------------------------------------------------------------

# Get set 1 data 

set1_df = @chain calibration_raw begin
    @filter(set == 1)
end

# Unique values of the GameID variable

@chain set1_df begin
    @select(gameid) 
    @distinct()
    @arrange(gameid)
end

# Experiment 1 data ----------------------------------------------------------------

# Get experiment 1 data (sets 5 and 6)

exp1_df = @chain calibration_raw begin
    @filter(set in (5, 6))
end

@glimpse(exp1_df)

# Unique values of the set 

@chain exp1_df begin
    @select(set) 
    @distinct()
    @arrange(set)
end

# Unique values of the GameID variable

@chain exp1_df begin
    @select(gameid) 
    @distinct()
    @arrange(gameid)
end

# Find the ambiguous B 

@chain exp1_df begin
    @select(amb) 
    @distinct()
end

# Number of lottery outcomes in option B for Experiment B

@chain exp1_df begin
    @select(lotnumb) 
    @distinct()
end

# Potential shapes for LotShapeA

@chain exp1_df begin
    @select(lotshapea) 
    @distinct()
end

# Potential shapes for LotShapeB

@chain exp1_df begin
    @select(lotshapeb) 
    @distinct()
end

# Button

@chain exp1_df begin
    @select(button) 
    @distinct()
end

# Payoffs

@chain exp1_df begin
    @select(payoff) 
    @distinct()
end

# Forgone 

@chain exp1_df begin
    @select(forgone) 
    @distinct()
end

# La and Lb, Ha, Hb

@chain exp1_df begin
    @select(la, lb, ha, hb) 
    @distinct()
end

# pHa, pHb

@chain exp1_df begin
    @select(p_ha, p_hb) 
    @distinct()
end

# Feedback

@chain exp1_df begin
    @select(feedback) 
    @distinct()
end

# LotNumA, LotNumB

@chain exp1_df begin
    @select(lotnuma, lotnumb) 
    @distinct()
end