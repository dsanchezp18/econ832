# Problem Set 3
# SFU ECON832 Computational Methods for Economics
# Spring 2024

# Problem 1 -------------------------------------------------------------------

# To change the utility function which describes preferences of DMs, we modify the U object in the replication code. 

#This function computes the smallest number of observations per menu (needed for τ_n)
## X=is a dataset
## There are 32 menus
function nintaun(X)
    dM=maximum(X[:,1])
    Ntau=10000.0
    for i=2:dM # First menu is empty
      Ntau=minimum([Ntau,sum(X[:,1].==i)])
    end
    return Ntau
end

# Finding nonredundant indices
function gindexes(Menus,dYu)
    dM=length(Menus)
    MM=zeros(dM,dYu)
    if dYu==6
        #For RUM we add the default to every menu
        Menus=[vcat(1,Menus[i].+1) for i in eachindex(Menus)]
        Menus[1]=[]
    end
    for i in 1:dM
        MM[i,Menus[i][1:end-1]].=1.0 # The last option in the menu is dropped
    end
    # Nonzero and linearly independent frequencies in calibrated probabilities
    gindex=findall(MM.==1.0)
    return gindex
end

# Given data computes empirical frequence for all menus and options
function frequencies(data)
    dM=maximum(data[:,1]); dY=maximum(data[:,2])-minimum(data[:,2])+1;
    F=zeros(Float64,dM,dY)
    for i in 1:size(data,1)
        F[data[i,1],data[i,2]+1]= F[data[i,1],data[i,2]+1]+1.0
    end
    # Computing sample frequencies
    P=zeros(Float64,dM,dY)
    P[1,1]=1.0
    P[2:end,:]=F[2:end,:]./sum(F[2:end,:],dims=2)
    # Throwing away P(default) since it is equal to 1-sum(P[1:5])
    P=P[:,2:end]
    return P
end

# This function translates integer menu identifiers to actual menus
function int2menu(number, menus=Menus)
    return menus[number]
end

# This function translates actual menus to integer menu identifiers
function menu2int(vector::Array{Int64,1})
    p=true; i=1;
    while p
        if int2menu(i)==vector
            p=false
        end
        i=i+1
    end
    return i-1
end

# This function computes all subsets of a given set
# Uses Combinatorics package
function subsetsint(intset,menus=Menus)
    Temp=collect(powerset(int2menu(intset, menus)))
    return [menu2int(Temp[i]) for i in 1:length(Temp)]
end

# Compute m_A(D) given the matrix of frequencies ps
function m(D,A,ps)
    if A==1
        return "error"
    end
    if ~issubset(D,subsetsint(A))
        return 0.0
    else
        etavecT=etavec_con(ps) #Computing the attention index
        if model=="LA" # LA
            return etavecT[D]/sum(etavecT[subsetsint(A)])
        elseif model=="EBA"
            beta=0.0   # EBA
            for i in 1:length(etavecT)
                if intersect(int2menu(i),int2menu(A))==int2menu(D)
                    beta=beta+etavecT[i]
                end
            end
            return beta
        end
    end
end

# Compute the atention index eta(D) for all D given the matrix of frequencies ps
# This function uses the direct formula from the paper
function etavec(ps)
    # Compting P(default)
    po=1.0 .- sum(ps,dims=2)
    DD=subsetsint(32) # All subsets
    etavec1=zeros(Float64,length(DD))
    for i=1:length(DD)
        BB=subsetsint(DD[i]) # Subsets of a given set
        Dt=int2menu(DD[i])
        betatemp=0.0
        for j=1:length(BB)
            betatemp=betatemp+elmobinv(Dt,BB[j],po) #adding up elements of inversion
        end
        etavec1[i]=betatemp
    end
    return etavec1
end

# This function computes the elements of the summation in the defintion if
# the atention index eta is directly computed
function elmobinv(Dt,BB,po)
    if model=="LA"
        return (-1.0)^(length(setdiff(Dt,int2menu(BB))))*po[32]/po[BB]
    elseif model=="EBA"
        return (-1.0)^(length(setdiff(Dt,int2menu(BB))))*po[menu2int(setdiff(vec(1:5),int2menu(BB)))]
    end
end


# Compute the constrained atention index eta(D) for all D given the matrix of frequencies ps
# This function restricts etas to be probabilities
function etavec_con(ps)
    # Computing P(default)
    po=1.0 .- sum(ps,dims=2)
    DD=subsetsint(32) # All subsets
    etamin=Model(Ipopt.Optimizer)
    ## commenting next line because of Ipopt
    #set_optimizer_attribute(etamin,"outlev",0)
    @variable(etamin, etaparam[1:length(DD)]>=0)
    @constraint(etamin, addm, sum(etaparam[t] for t in 1:length(DD))==1)
    if model=="LA"
        ## p(o,X)/p(o,A)
        @objective(etamin,Min,sum((po[32]/po[DD[i]]-sum(etaparam[l] for l in subsetsint(DD[i])))^2 for i in 1:length(DD)))
    elseif model=="EBA"
        ## p(o,X-A)
        @objective(etamin,Min,sum((po[menu2int(setdiff(vec(1:5),int2menu(DD[i])))]-sum(etaparam[l] for l in subsetsint(DD[i])))^2 for i in 1:length(DD)))
    end

    JuMP.optimize!(etamin)

    return value.(etaparam)
end


# Computing the constrained calibrated p_\pi from the matrix of frequencies
# This function is constrained to return probabilities
function pipie_cons(x)
    dM,dYu=size(x)
    MM=[ m(D,A,x) for D in 1:dM, A in 2:dM] # Matrix of consideration probabilites
    ConsB=(x.>0.0) # Matrix of 0/1. ConsB=0 if P(a in A)=0 and =1 otherwise
    pipiemin=Model(Ipopt.Optimizer)
    ##commenting next line because of Ipopt
    #set_optimizer_attribute(pipiemin,"outlev",0)
    @variable(pipiemin, pipieparam[1:dM,1:dYu]>=0)
    @constraint(pipiemin, sump[l=2:dM], sum(pipieparam[l,t] for t in 1:dYu)==1)
    @constraint(pipiemin, [l=1:dM,t=1:dYu], pipieparam[l,t]<=ConsB[l,t])

    @objective(pipiemin,Min, sum([(x[A,a]-sum(MM[D,A]*pipieparam[D,a] for D in 1:dM))^2 for A in 1:dM-1, a in 1:dYu]))
    JuMP.optimize!(pipiemin)
    
    return value.(pipieparam)
end


# Computing G
# U is the set of preferences
# M is the set of menus
# gindexsh are coordinates of nonzero linearly independent p_pi
function matrixcons(gindexsh, M, U)
    dYu=length(U[1])
    dYm=length(M)
    if dYu==6
        M=[vcat(1,M[i].+1) for i in eachindex(M)]
        M[1]=[]
    end
    d=length(U)
    d2=length(gindexsh)
    B=zeros(d2,d)
    m1=1
    for j in 1:d
        pp=zeros(dYm,dYu)
        for i in eachindex(M)
            if length(M[i])>0 # Skipping empty menu
                for k in 1:dYu
                    if k==M[i][argmax(U[j][M[i]])]
                        pp[i,k]=1.0
                    end
                end
            end
        end
        B[:,m1]=pp[gindexsh] #picking only relevant elements, indexes that are not always zero
        m1=m1+1
    end
    if dYu==6 # If model is RUM then return B
        return B
    else #otherwise return
        return [[B zeros(size(B,1),dYm)];[zeros(dYm,size(B,2),) I]]
    end
end

# This function computes the vector of linearly independent nonzero elements of
# p_\pi and eta (if needed)
function estimateg(X,gindex)
  Y=frequencies(X)
  if length(gindex)==80 # If model is RUM
       return [1.0 .- sum(Y,dims=2) Y][gindex]
   else # if model is LA or EBA
       return vcat(pipie_cons(Y)[gindex],etavec(Y))
   end
end

# This function computes the test statistic
function kstesstat(ghat,G,Omegadiag,taun,solution)
    if sum(isnan.(ghat))>0
        return -100
    end
    dr,dg=size(G)
    KS=Model(Ipopt.Optimizer)
    ##commenting next line because of Ipopt 
    ##set_optimizer_attribute(KS,"outlev",0)
    @variable(KS,etavar[1:dg]>=taun/dg) #taun is a tuning parameter
    @objective(KS,Min,sum((sqrt(Omegadiag[r])*(ghat[r]-sum(G[r,l]*etavar[l] for l in 1:dg)))^2 for r in 1:dr))
    JuMP.optimize!(KS)
    if solution==true
        return G*value.(etavar)
    else
        return objective_value(KS)
    end
end

# This function computes the bootstrap statistic given the seed for a given frame
function ksbootseed(seed)
  ghatb=estimateg(genbootsample(X,seed),gindex)
  return kstesstat(ghatb-ghat+etahat,G,Omegadiag,taun,false)
end


# This function computes the bootstrap statistic given the seed for the model
# with stabel preferences
function ksbootseedstable(seed)
    ghat1b=estimateg(genbootsample(X1,seed),gindex)
    ghat2b=estimateg(genbootsample(X2,seed^2),gindex)
    ghat3b=estimateg(genbootsample(X3,2*seed+5),gindex)
    ghatb=vcat(ghat1b,ghat2b,ghat3b)
    return kstesstat(ghatb-ghat+etahat,G,Omegadiag,taun,false)
end

#This function generates a bootstrap sample that has positive probability of the outside option for every menu
function genbootsample(Xt,seed)
    rng1=MersenneTwister(seed)
    dd=false
    Xtb=zeros(size(Xt))
    while dd==false
        Xtb=Xt[rand(rng1,1:size(Xt,1),size(Xt,1)),:]
        dd=minimum(1.0 .- sum(frequencies(Xtb),dims=2))>0.0
    end
    return Xtb
end 

function preferences(dYu)
  U=collect(permutations(vec(1:dYu))) # All preference orders
    return U
end

## This code was created by Victor H. Aguiar and Nail Kashaev
## The code is part of the paper "Random Utility and Limited Consideration" by Aguiar, Boccardi, Kashaev, and Kim (ABKK) QE, 2022
## This version is modified to work with Ipopt instead of KNITRO
## The code is written in Julia 1.6.4

## This part requires us to use the Pkg package to install the necessary packages and to keep reproducibility, 
## to obtain the exact numbers as in ABKK you need a KNITRO license, you can get it from Artelys. 
using Pkg
## These lines instantiate the packages and activate the environment, KNITRO is not necessary for this part
Pkg.activate(".")
Pkg.instantiate()
using Distributed
using Statistics
using DataFrames, CSV
## This part is to run the code in parallel
addprocs(7)

@everywhere begin
  using Random
  using Combinatorics
  using LinearAlgebra
  using JuMP
  using Ipopt
end

#@everywhere model=$(ARGS[1])   # put "LA", "EBA", or "RUM"
## This line is not necessary if you are running the code from a shell, but if you are running it from a IDE you need to put the model as an argument
@everywhere model="RUM"   # put "LA", "EBA", or "RUM"
println(model)

## Defining the file directories
tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("Replication_RULC-main",tempdir1)[end]]

## Functions
@everywhere include($(rootdir)*"/main/functions_common_testing.jl")

## Common parameters
dYm=5                               # Number of varying options in menus
model=="RUM" ? dYu=6 : dYu=5        # For RUM there are 6 options instead of 5
Menus=collect(powerset(vec(1:dYm))) # Menus
gindex=gindexes(Menus,dYu)          # Indices that correspond to nonzero linearly independent frequencies
U=preferences(dYu)                  # All preference orders
G=matrixcons(gindex, Menus, U)      # Marix of 0 and 1
if model=="RUM"
  G=vcat(G,G,G)                       # To impose stability we need to repeat G for every frame for RUM 
else 
  B=G[1:size(G,1)-length(Menus),1:size(G,2)-length(Menus)]
  Bt=[B zeros(size(B,1),3*length(Menus))]
  G=[Bt;
  zeros(length(Menus), size(B,2)) I zeros(length(Menus), 2*length(Menus));
   Bt;
  zeros(length(Menus), length(Menus)+size(B,2)) I zeros(length(Menus), length(Menus));
   Bt;
   zeros(length(Menus), 2*length(Menus)+size(B,2)) I ]
end

Omegadiag=ones(size(G,1))           # Vector of weigts

## Data for all 3 frames
X1=Matrix(CSV.read(rootdir*"/data/menu_choice_high.csv", DataFrame))
X2=Matrix(CSV.read(rootdir*"/data/menu_choice_medium.csv", DataFrame))
X3=Matrix(CSV.read(rootdir*"/data/menu_choice_low.csv", DataFrame))
println("Data is ready!")
# Sample sizes
N1=size(X1,1)
N2=size(X2,1)
N3=size(X3,1)
# Smallest sample size
N=minimum([N1,N2,N3])
# Smallest sample per menu
Ntau=nintaun(X1)
# Tuning parameter as suggested in KS
taun=sqrt(log(Ntau)/Ntau)

## Testing
println("Testing...")
# Estimates of g for all 3 frames
ghat1=estimateg(X1,gindex)
ghat2=estimateg(X2,gindex)
ghat3=estimateg(X3,gindex)
ghat=vcat(ghat1,ghat2,ghat3)
etahat=kstesstat(ghat,G,Omegadiag,taun,true)
@everywhere begin
  X1=$X1; N1=$N1; X2=$X2; N2=$N2; X3=$X3; N3=$N3; N=$N
  gindex=$gindex; ghat=$ghat; G=$G;
  Omegadiag=$Omegadiag; taun=$taun; etahat=$etahat; Menus=$Menus
end
# Test statistic
Tn=N*kstesstat(ghat,G,Omegadiag,0.0,false)
# Bootstrap statistics
@time Boot=pmap(ksbootseedstable,1:1000) # use ksbootseedstable function
# Pvalue. If it is 0.0, then pvalue<0.001
pvalue=mean(Tn.<=N.*collect(Boot))

## Saving the output
CSV.write(rootdir*"/main/results/Tn_pval_$(model)_stable.csv", DataFrame(Tn=Tn, pvalue=pvalue))
println("Done!")

# Problem 2 ---------------------------------------------------------------------

# Use bootstrapping to provide boostrap standard errors for β/b in the supply and demand model. 
# Must use a defined function for the 2SLS estimation and then use the function to perform the bootstrapping with pmap. 

## Preliminaries --------------------------------------------------------------

# Import packages

import JuMP 
import Ipopt 
using Random 
using DataFrames
using Distributions 
using Distributed
using Statistics

# Predefine parameters

Random.seed!(78909434) 

N = 100
a = 0.5 # Demand intercept
b = 0.5 # Demand slope (note that we define as positive because in the demand equation there will be a negative sign attached to it)
α = 0.5 # Supply intercept
β = 0.5 # Supply slope

σ_u = 1.5 # Std of demand shocks
σ_v = 2.5 # Std of supply shocks

μ_u = 0.0 # Mean of demand shocks
μ_v = 0.0 # Mean of supply shocks

## Simulation of observational data -------------------------------------------

# u and v (random supply and demand shocks)

u = zeros(N)
v = zeros(N) 

for i in 1:N
    u[i] = rand(Normal(μ_u, σ_u))
    v[i] = rand(Normal(μ_v, σ_v))
end

# price vector p

p = zeros(N)

for i in 1:N
    p[i]=((a-α)/(β+b)) + ((u[i]-v[i])/(β+b))
end

# quantity in the market y

y = zeros(N)

for i in 1:N
    y[i]= ((a*β + b*α)/(b+β)) + ((β*u[i] + b*v[i])/(β+b))
end

# D and S vectors (quantities)

D = zeros(N)
S = zeros(N)

for i in 1:N
    D[i] = a - b*p[i] + u[i] 
    S[i] = α + β*p[i] + v[i]
end

# Relationships of instruments with random shocks

c_u = 0.5
c_v = 0.5

# Standard deviations of the unobserved parts of random shocks

σ_ϵu = 0.25
σ_ϵv = 0.5

# Means of the unobserved parts of random shocks

μ_ϵ_u = 0.0
μ_ϵ_v = 0.0

# Error of the instruments

ϵ_u = zeros(N)
ϵ_v = zeros(N)

for i in 1:N
    ϵ_u[i] = rand(Normal(μ_ϵ_u, σ_ϵu)) 
    ϵ_v[i] = rand(Normal(μ_ϵ_v, σ_ϵv))
end 

x_u = zeros(N)
x_v = zeros(N)

for i in 1:N
    x_u[i] = (( u[i] - ϵ_u[i] ) / (c_u) )
    x_v[i] = (( v[i] - ϵ_v[i] ) / (c_v)) 
end

## Standard errors ------------------------------------------------------------

# I will need a dataframe with the data that I have (my sample) to perform the bootstrapping

df = DataFrame(p = p, y = y, x_v = x_v)

# Now I will perform the bootstrapping

B = 10000 # Number of bootstrap samples

# Add worker processes for parallel computing
addprocs(4)  

@everywhere using JuMP
@everywhere using Ipopt 
@everywhere using DataFrames, Statistics, Distributions

@everywhere begin
    function two_stage_least_squares(df)
    N = 100
        # First stage
        first_stage = JuMP.Model(Ipopt.Optimizer)
        JuMP.@variable(first_stage, π0) # Pi 0 is the intercept of the first stage regression
        JuMP.@variable(first_stage, π1) # Pi 1 is the slope of the first stage regression
        JuMP.@objective(first_stage, Min, sum((df.p[i] - π0 - π1*df.x_v[i])^2  for i in 1:N))
        JuMP.optimize!(first_stage) # Perform the optimization
        π0_hat = JuMP.value.(π0) # Access the intercept
        π1_hat = JuMP.value.(π1) # Access the slope
        p_hat = zeros(N) # Create a vector of zeros to later allocate values
        for i in 1:N
            p_hat[i] = π0_hat + π1_hat*df.x_v[i]
        end
        # Second stage
        second_stage = JuMP.Model(Ipopt.Optimizer)
        JuMP.@variable(second_stage, γ0) # Gamma 0 is the intercept of the second stage regression
        JuMP.@variable(second_stage, γ1) # Gamma 1 is the slope of the second stage regression
        JuMP.@objective(second_stage, Min, sum((df.y[i] - γ0 - γ1*p_hat[i])^2  for i in 1:N)) # Minimize the sum of squared residuals. This is the objective function
        JuMP.optimize!(second_stage) # Perform the optimization
        γ1_hat = -JuMP.value.(γ1) # Access the slope of the second stage regression
        return γ1_hat
    end

    function bootstrap_sample(df)
        sample = df[rand(1:size(df, 1), size(df, 1)), :]  # Bootstrap sample
        return two_stage_least_squares(sample)
    end
end

# Assume `df` is your data
bootstrap_estimates = pmap(bootstrap_sample, [df for _ in 1:B])

# Calculate standard error
bootstrap_se = std(bootstrap_estimates)

println("Bootstrap standard error: ", bootstrap_se)