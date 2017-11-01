module SemiparametricAUC

# package code goes here
using Distributions
using DataStructures
using Iterators
using DataArrays
using DataFrames
import Base: log, mean, var, zeros, length, isa, values, sort, inv, diagm, sqrt, diag,hcat

include("calculate-auc.jl")
include("sauc.jl")
include("simulation-one-predtictor.jl")

export calculate_auc

"""
  calculate_auc(ya, yb)

  This function takes two DataArrays `ya` and `yb` and calculates variance of predicted AUC,
  logit of predicted AUC, and variance of logit of predicted AUC responses passed.

"""
calculate_auc(ya = DataFrames.DataArray([2,3,4,3]), yb = DataFrames.DataArray([3,2,1,3,4,3]))

export semiparametricAUC

"""
semiparametricAUC(model_formula = y ~ x1, treatment_group = :group, data = fasd)

  This function is used to fit semiparametric AUC regression model specified by
  giving a formula object of response and covariates and a separate argument of treatment
  group. It will convert variables other than response into factors, estimate model parameters,
  and display results.

"""
SemiparametricAUC.semiparametricAUC(model_formula = y ~ x1, treatment_group = :group, data = fasd)

export calculate_auc_simulation

"""
  calculate_auc_simulation(ya, yb)

  This function takes two DataArrays `ya` and `yb` and calculates variance of predicted AUC,
  logit of predicted AUC, and variance of logit of predicted AUC responses passed.

"""

calculate_auc_simulation(ya = DataFrames.DataArray([2,3,4,3]), yb = DataFrames.DataArray([3,2,1,3,4,3]))


export simulate_one_predictor


"""
  simulate_one_predictor(iter, m, p)

  It asks for number of iterations to be run, number of observations in treatment
  and control groups for the simulation of Semiparametric AUC regression adjusting for one discrete
  covariate. In this simulation, true model parameters are as follows: \beta_0 = 0.15,
  \beta_1 = 0.50, \beta_2 = 1.
"""

simulate_one_predictor(;iter = 500, m = 100, p = 120)

end # module
