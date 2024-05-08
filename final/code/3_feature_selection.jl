# SFU ECON832 Final
# Spring 2024
# Analysis of feature selection for the final project

# Preliminaries -------------------------------------------------------------------

# Activate the package environment

using Pkg

Pkg.activate("final_env")

# Load packages

using TidierData
using DataFrames
using TidierFiles
using TidierPlots
using HypothesisTests
using Plots

# Load data

# Cleaned calibration data with all features

calibration_cleaned = @chain begin
        read_csv("data/output/cleaned_calibration.csv",
                 missingstring = ["NA"])
        @clean_names()
end

# Cleaned experiment 1 data with all features

df_exp1 = @chain begin
    read_csv("data/output/df_exp1.csv",
             missingstring = ["NA"])
    @clean_names()
end

# Data preparation -------------------------------------------------------------------

# Quick data preparation for the feature selection analysis

df = @chain df_exp1 begin
    @mutate(choice = if_else(b == 1, "B", "A"))
end 

# Descriptive statistics -------------------------------------------------------------------

# Get the percentage of choices for each lottery -- this will be used to determine the baseline accuracy

df_choice = @chain df begin
    @group_by(choice)
    @summarize(number = n())
    @mutate(percentage = number / sum(number) * 100)
    @ungroup()
end

# Get the percentage within each location

df_choice_by_set = @chain df begin
    @group_by(location_rehovot, choice)
    @summarize(number = n())
    @mutate(percentage = number / sum(number) * 100)
    @ungroup()
end

# Get the percentage by gender

df_choice_by_gender = @chain df begin
    @group_by(gender, choice)
    @summarize(number = n())
    @mutate(percentage = number / sum(number) * 100)
    @ungroup()
end

# Percentages by set 

df_choice_by_set = @chain df begin
    @group_by(choice, set)
    @summarize(number = n())
    @mutate(percentage = number / sum(number) * 100)
    @ungroup()
end

# Analysis of risk features -------------------------------------------------------------------

# Correlate risk features with the outcome variable to observe the relationship and justify their inclusion in the DFNN

## Expected values ---------------------------------------------------------------

# Correlate expected values with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.hb, df.b)

cor(df.ha, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.hb, df.b)) # Not statistically significant

pvalue(CorrelationTest(df.ha, df.b)) # Statistically significant

# Do a scatter plot of the risk features against the outcome variable (using TidierPlots)

ggplot(df) +
    geom_boxplot(@aes(x = choice, y = hb)) +
    labs(title = "Boxplot of E[B] against Having chosen B",
         x = "Choice of lottery B or A",
         y = "E[B]")

# No particular preference if we don't control for any other variables. No reason to include these features in the DFNN unless they work with other variables.

# Number of outcomes of the lotteries ---------------------------------------------------------------

# Correlate number of lottery outcomes with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.lotnumb, df.b) # Stronger

cor(df.lotnuma, df.b) # Negative, small number

# Perform correlation test 

pvalue(CorrelationTest(df.lotnumb, df.b)) # Statistically significant

pvalue(CorrelationTest(df.lotnuma, df.b)) # Statistically significant

# Do a scatter plot of the number of lottery outcomes of b against the outcome variable (using TidierPlots)

lotnumb_boxplot =
    ggplot(df) +
    geom_boxplot(@aes(x = choice, y = lotnumb)) +
    labs(title = "Choice of lottery against number of B lottery outcomes",
         x = "Choice of lottery B or A",
         y = "Number of Outcomes of B")

lotnumb_boxplot

ggsave(lotnumb_boxplot, "figures/lotnumb_boxplot.png", scale = 1.5)

# Correlation between lotteries ---------------------------------------------------------------

# Correlate the correlation between lotteries with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.corr, df.b) 

# Perform correlation test

pvalue(CorrelationTest(df.corr, df.b)) # Statistically significant

# Do a scatter plot of the correlation between lotteries against the outcome variable (using TidierPlots)

obs_by_corr =  @chain df begin
        @group_by(choice, corr)
        @summarize(number = n())
        @mutate(corr = as_string(corr))
        @ungroup()
end

ggplot(obs_by_corr) +
    geom_col(@aes(x = corr, y = number, colour = choice, group = choice), position = "dodge", stat = "identity") +
    labs(title = "Number of Observations by Correlation between Lotteries",
         x = "Correlation between Lotteries",
         y = "Number of Observations") +
    scale_colour_manual(values = ["#F8766D", "#00BFC4"])

## Ambiguity -------------------------------------------------------------------

# Correlate ambiguity with the outcome variable to observe the relationship and justify its inclusion in the DFNN

cor(df.amb, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.amb, df.b)) # Not statistically significant

## Probabilities -------------------------------------------------------------------

# Correlate probabilities with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.p_hb, df.b)

cor(df.p_ha, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.p_hb, df.b)) # Not statistically significant

pvalue(CorrelationTest(df.p_ha, df.b)) # Not statistically significant

## Low payoffs -------------------------------------------------------------------

# Correlate low payoffs with the outcome variable to observe the relationship and justify their inclusion in the DFNN

cor(df.lb, df.b)

cor(df.la, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.lb, df.b)) # Statistically significant

pvalue(CorrelationTest(df.la, df.b)) # Statistically significant

# Do a scatter plot of the low payoffs against the outcome variable (using TidierPlots)

ggplot(df) +
    geom_boxplot(@aes(x = choice, y = lb)) +
    labs(title = "Boxplot of Low Payoffs of B against Having chosen B",
         x = "Choice of lottery B or A",
         y = "Low Payoffs of B")

## Shape of the distribution -------------------------------------------------------------------

# Bar plot of the shape of the distribution of the B lottery against the outcome variable

df_by_lotshape  = 
@chain df begin
    @group_by(choice, lotshapeb)
    @summarize(number = n())
    @ungroup()
end

barplot_by_lotshape =
ggplot(df_by_lotshape) +
    geom_col(@aes(x = lotshapeb, y = number, colour = choice), position = "dodge") + 
    labs(x = "Shape of the Distribution of B",
         y = "Number of Observations") +
    scale_colour_manual(values = ["#F8766D", "#00BFC4"])

barplot_by_lotshape

ggsave(barplot_by_lotshape, "figures/barplot_by_lotshape.png", scale = 1.2)

# A symmetric distribution greatly favours B

## Payoff features -------------------------------------------------------------------

# Correlate the payoff the subject got from the the choice in the current trial

cor(df.payoff, df.payoff)

CorrelationTest(df.b, df.payoff)

# The Payoff provided by B

cor(df.b, df.bpay)

CorrelationTest(df.b, df.bpay)

# The Payoff provided by A

cor(df.b, df.apay)

CorrelationTest(df.b, df.apay)

# Do a boxplot of the payoff obtained by the subject against the choice of lottery

payoff_boxplot = 
ggplot(df) +
    geom_boxplot(@aes(x = choice, y = payoff)) +
    labs(title = "Boxplot of Payoff against Choice of Lottery",
         x = "Choice of lottery B or A",
         y = "Payoff")

payoff_boxplot

# Payoff of b against the outcome variable

payoff_b_boxplot =
ggplot(df) +
    geom_boxplot(@aes(x = choice, y = bpay)) +
    labs(title = "Boxplot of Payoff of B against Having chosen B",
         x = "Choice of lottery B or A",
         y = "Payoff of B")

payoff_b_boxplot

# Attention features -------------------------------------------------------------------

## Order of the lotteries -------------------------------------------------------------------

# Correlate order of lotteries with the outcome variable to observe the relationship and justify its inclusion in the DFNN

cor(df.order, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.order, df.b)) # Not statistically significant

# Not statistically significant

## Forgone -------------------------------------------------------------------

# Correlate the forgone lottery with the outcome variable to observe the relationship and justify its inclusion in the DFNN

cor(df.forgone, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.forgone, df.b)) # Statistically significant

# Do a boxplot 

forgone_boxplot =
    ggplot(df) +
    geom_boxplot(@aes(x = choice, y = forgone)) +
    labs(title = "Boxplot of Forgone against Choice of Lottery",
         x = "Choice of lottery B or A",
         y = "Forgone")

forgone_boxplot

# Forgone payoff favours A

## Reaction time -------------------------------------------------------------------

# Filter out missing values in reaction time

df_with_rt = @chain df begin
    @filter(!ismissing(rt))
end

# Correlate reaction time with the outcome variable to observe the relationship and justify its inclusion in the DFNN

cor(df_with_rt.rt, df_with_rt.b)

# Perform correlation test

pvalue(CorrelationTest(df_with_rt.rt, df_with_rt.b)) # Not statistically significant

## Feedback -------------------------------------------------------------------

# Bar plot comparing choices with feedback and without feedback

choices_by_feedback = @chain df begin
    @mutate(feedback = if_else(feedback == 1, "No Feedback", "Feedback"))
    @group_by(choice, feedback)
    @summarize(number = n())
    @ungroup()
end

barplot_by_feedback =
    ggplot(choices_by_feedback) +
    geom_col(@aes(x = choice, y = number, colour = feedback), position = "dodge") +
    labs(title = "Number of Choices by Feedback",
         x = "Choice of lottery B or A",
         y = "Number of Observations") +
    scale_colour_manual(values = ["#F8766D", "#00BFC4"])

barplot_by_feedback

# Feedback greatly favours A

ggsave(barplot_by_feedback, "figures/barplot_by_feedback.png", scale = 1.2)

# Blocks -------------------------------------------------------------------

# Correlation of time blocks with the outcome variable

cor(df.block, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.block, df.b)) # Not statistically significant

# Button -------------------------------------------------------------------

# Correlation of button with the outcome variable

cor(df.button_r, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.button_r, df.b)) # Statistically significant

# Column plot

choices_by_button = @chain df begin
    @mutate(button_r = if_else(button_r == 1, "Button R", "Button L"))
    @group_by(choice, button_r)
    @summarize(number = n())
    @ungroup()
end

columnplot_by_button =
    ggplot(choices_by_button) +
    geom_col(@aes(x = choice, y = number, colour = button_r), position = "dodge") +
    labs(title = "Number of Choices by Button",
         x = "Choice of lottery B or A",
         y = "Number of Observations") +
    scale_colour_manual(values = ["#F8766D", "#00BFC4"])

## Trial -------------------------------------------------------------------

# Correlation of trial with the outcome variable

cor(df.trial, df.b)

# Perform correlation test

pvalue(CorrelationTest(df.trial, df.b)) # Not statistically significant
