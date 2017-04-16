module SemiparametricAUC

# package code goes here
using Distributions
using Iterators
using DataArrays
using DataFrames
import Base: log, mean, var, zeros, length, isa, values, sort, inv, diagm, sqrt, diag,hcat

include("calculate-auc.jl")
include("sAUC.jl")
include("simulation-one-predtictor.jl")

export calculate_auc
export semiparametricAUC
export calculate_auc_simulation
export simulate_one_predictor

end # module
