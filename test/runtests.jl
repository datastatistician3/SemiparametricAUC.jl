using SemiparametricAUC
using DataArrays
using Base.Test

# write your own tests here
@test 1 == 1

println("Testing Som's Package")
include("calculate-auc.jl")

@test calculate_auc(ya = DataArray([2,3]), yb = DataArray([3,2,1])) == (1.5468749999999993,0.6931471805599452)
