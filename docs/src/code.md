# SemiparametricAUC.jl Code

## calculate_auc

```julia
"""
  calculate_auc(ya, yb)

  # "Calcualtes AUC related estimates"

  This function takes two DataArray arguments `ya` and `yb` and calculates variance of predicted AUC,
  logit of predicted AUC, and variance of logit of predicted AUC responses passed.
"""
calculate_auc(ya = DataFrames.DataArray([2,3,4,3]), yb = DataFrames.DataArray([3,2,1,3,4,3]))

function calculate_auc(; ya::DataArrays.DataArray = nothing, yb::DataArrays.DataArray = nothing)
  m = length(ya)
  p = length(yb)
  I = zeros(m, p)
    for i in range(1,m)
        for j in range(1,p)
            if ya[i] > yb[j]
              I[i,j] = 1
            elseif ya[i] == yb[j]
              I[i,j] = 0.5
            else
               I[i,j] = 0
            end
        end
    end
    finv(x::Float64) = return(-log((1/x)-1))
    auchat = mean(I)
    finvhat = finv(auchat)
    vya = mean(I,2)
    vyb = mean(I,1)
    svarya = var(vya)
    svaryb = var(vyb)
    vhat_auchat = (svarya/m) + (svaryb/p)
    v_finv_auchat = vhat_auchat/((auchat^2)*(1-auchat)^2)
    logitauchat = log(auchat/(1-auchat))
    var_logitauchat = vhat_auchat /((auchat^2)*(1-auchat)^2)
    return((var_logitauchat, logitauchat))
end
```

#### Example

```julia
calculate_auc(ya = DataFrames.DataArray([2,3,4,5]), yb = DataFrames.DataArray([2,3,4,5]))
```

## semiparametricAUC

```julia
"""
  semiparametricAUC(model_formula, treatment_group, data)

  This function is used to fit semiparametric AUC regression model specified by
  giving a formula object of response and covariates and a separate argument of treatment
  group. It will convert variables other than response into factors, estimate model parameters,
  and display results.

  This function takes a formula object `model_formula` with response and covariates such as {response ~ x1 + x2},
  group argument `treatment_group`, which is treatment group for which a comparision is to be made, and
  a data argument `data` which is a DataFrame object that contains variables needed for the analysis.
"""
semiparametricAUC(model_formula = y ~ x1, treatment_group = :group, data = fasd)

function semiparametricAUC(; model_formula::DataFrames.Formula = throw(ArgumentError("Argument model_formula is missing")),
  treatment_group::Symbol = throw(ArgumentError("Argument treatment_group is missing")),
    data::DataFrames.DataFrame = throw(ArgumentError("Argument data is missing")))
  # fasd = DataFrames.readtable(joinpath(Pkg.dir("SemiparametricAUC"), "data/fasd.csv"))
  # model_formula = y ~ x1 + x2
  # treatment_group = :group
  # data = fasd

  if (isa(model_formula, Formula))
    input_covariates = DataFrames.Terms(model_formula).terms
    n1 = length(input_covariates)
    input_response = Symbol(DataFrames.Terms(model_formula).eterms[1])
  else error("Please put response and input as DataFrames.Formula object. For example, model_formula = response ~ x1 + x2")
  end

  if (!isa(treatment_group, Symbol))
    error("The parameter treatment_group should be Symbol object. For e.g. :x1")
  end

  if (!isa(data, DataFrames.DataFrame))
    error("The parameter data should be DataFrames.DataFrame object.")
  end
  input_treatment = Symbol(treatment_group)
  group_covariates = vcat(input_covariates, input_treatment)

  # if (sum([isa(data[:,i], DataFrames.PooledDataArray) for i in group_covariates]) == 0)
  #   error("Please put response and input as formula. For example, response ~ x1 + x2")
  # else
  #     println("Great")
  # end

  print_with_color(:red,"Data are being analyzed. Please, be patient.\n\n")
  # split by factors
  # TO-DO: make sure that oerder of the variables are aligned with order of coefnames(mf)
  ds = data[vcat(input_response,Symbol.(group_covariates))]
  grouped_d = DataFrames.groupby(ds, group_covariates)

  half_data = 0.5
  set1 =  Dict()
  for i in 1:2:length(grouped_d)
    set1[i] = grouped_d[i]
  end
  set1_sorted = DataStructures.SortedDict(set1)

  set2 = Dict()
  for i in 2:2:length(grouped_d)
    set2[i] = grouped_d[i]
  end
  set2_sorted = DataStructures.SortedDict(set2)

  # TO-DO: make sure that order of the variables are aligned with order of coefnames(mf)
  logitauchat_matrix = collect(calculate_auc(ya = set1_sorted[i][:,input_response], yb = set2_sorted[i+1][:,input_response])
    for i in 1:2:length(grouped_d))

  dff = DataFrames.DataFrame([y[i] for y in logitauchat_matrix, i in 1:length(logitauchat_matrix[1])])
  var_logitauchat = dff[1]
  gamma1 = dff[2]

  # Change to adjust more than 2 d array
  # get levels
  dict_levels = Dict()
  for i in 1:length(input_covariates)
    dict_levels[i] = DataArrays.levels(ds[input_covariates[i]])
  end

  # for expand.grid
  matrix_x = collect(Iterators.product(values(sort(dict_levels))...))

  df_from_tuple = DataFrames.DataFrame([y[i] for y in matrix_x, i in 1:length(matrix_x[1])])
  df_from_tuple[input_response] = gamma1
  df_from_tuple[:var_logit] = var_logitauchat

  function convert_to_factor(x)
      return(DataFrames.pool(x))
  end

  for i in input_covariates
    df_from_tuple[i] = convert_to_factor(df_from_tuple[i])
  end

  # model.matrix using DataFrames (ModelMatrix)
  mf = ModelFrame(DataFrames.Terms(model_formula), df_from_tuple)
  mm = ModelMatrix(mf)

  coefnames(mf)
  Z = mm.m
  tau  =  diagm([1/i for i in var_logitauchat])
  ztauz = inv(Z' * tau * Z)
  var_betas = diag(ztauz)
  std_error = sqrt(var_betas)
  betas = ztauz * Z' * tau * gamma1

  threshold = Distributions.quantile(Distributions.Normal(), 0.975)

  lo = betas - threshold*std_error
  up = betas + threshold*std_error
  ci = hcat(betas,lo,up)

  return(SemiparametricAUC.coefs_table(mf = mf, lo = lo, up = up, betass = betas, std_errors = std_error))
end
```

#### Example
```julia
semiparametricAUC(model_formula = y ~ x1, treatment_group = :group, data = fasd)
```

## calculate_auc_simulation

```julia
"""
  calculate_auc_simulation(ya, yb)

  This function takes two DataArray arguments `ya` and `yb` and calculates variance of predicted AUC,
  logit of predicted AUC, and variance of logit of predicted AUC responses passed.
"""

calculate_auc_simulation(ya = DataFrames.DataArray([2,3,4,3]), yb = DataFrames.DataArray([3,2,1,3,4,3]))

function calculate_auc_simulation(; ya::Array = nothing, yb::Array = nothing)
  m = length(ya)
  p = length(yb)
  I = zeros(m, p)
    for i in range(1,m)
        for j in range(1,p)
            if ya[i] > yb[j]
              I[i,j] = 1
            elseif ya[i] == yb[j]
              I[i,j] = 0.5
            else
               I[i,j] = 0
            end
        end
    end
    finv(x::Float64) = return(-log((1/x)-1))
    auchat = mean(I)
    finvhat = finv(auchat)
    vya = mean(I,2)
    vyb = mean(I,1)
    svarya = var(vya)
    svaryb = var(vyb)
    vhat_auchat = (svarya/m) + (svaryb/p)
    v_finv_auchat = vhat_auchat/((auchat^2)*(1-auchat)^2)
    logitauchat = log(auchat/(1-auchat))
    var_logitauchat = vhat_auchat /((auchat^2)*(1-auchat)^2)
    return(auchat, finvhat, vhat_auchat)
end
```
#### Example
```julia
calculate_auc_simulation(ya = DataFrames.DataArray([2,3,4,3]), yb = DataFrames.DataArray([3,2,1,3,4,3]))
```

## simulate_one_predictor

```julia
"""
  simulate_one_predictor(iter, m, p)

  It asks for number of iterations `iter` to be run, number of observations `m` in treatment
  and control groups `p` for the simulation of Semiparametric AUC regression adjusting for one discrete
  covariate. In this simulation, true model parameters are as follows: β0 = 0.15, β1 = 0.50, β2 = 1.

"""
simulate_one_predictor(iter = 500, m = 100, p = 120)

function simulate_one_predictor(;iter = 500, m = 100, p = 120)
    iter = iter
    finvhat = gamma1 = Array(Float64, 3)
    AUChat = Array(Float64, 3)
    Vhat_auchat = Array(Float64, 3)
    lo = up = Array(Float64, iter, 3)
    d  = Array(Float64, m, 3)
    nd = Array(Float64, p, 3)

    m_betas = Array(Float64, iter, 3)
    sd_betas = Array(Float64, iter, 3)
    lower = upper = score =  cov_b = Array(Float64, iter, 3)
    v_finv_auchat = gamma = Array(Float64, 3)
    all_var = Array(Float64, 3)
    var_finv_auchat = Array(Float64, iter, 3)

    for z in range(1,iter)
      d1 = [0.0, 0.0, 0.0]
      d2 = [0, 0.50, 1.00]
      d0 = 0.15
      y = 1:3
      for k in y
        result = Array(Float64, k, 3)
        u1 = randexp(p) # rand(Exponential(1), p)
        u2 = randexp(m) # rand(Exponential(1), m)
        d[:,k]=-log(u2) + d0 +(d1[k] + d2[k])
        nd[:,k]=-log(u1) + d1[k]

        result= calculate_auc_simulation(ya = d[:,k], yb = nd[:,k])
        AUChat[k]=result[1]
        finvhat[k]=result[2]
        Vhat_auchat[k] = result[3]
        v_finv_auchat[k] = Vhat_auchat[k]/(((AUChat[k])^2)*(1-(AUChat[k]))^2)  #Variance of F inverse
      end
      gamma1 = finvhat
      Z = reshape([1,0,0,1,1,0,1,0,1], 3, 3)'

      tau  =  diagm([1/i for i in v_finv_auchat])

      ztauz = inv(Z' * tau * Z)
      var_betas = diag(ztauz)
      std_error = sqrt(var_betas)
      betas = ztauz * Z' * tau * gamma1

      m_betas[z,:]  =  betas
      var_finv_auchat[z,:] = var_betas
    end
    lo = m_betas .- 1.96*sqrt(var_finv_auchat)
    up = m_betas .+ 1.96*sqrt(var_finv_auchat)
    ci = hcat(lo,up)
    ci_betas = ci[:,[1,4,2,5,3,6]]
    return(m_betas, var_finv_auchat, ci_betas, iter)
end
```

#### Example
```julia
simulate_one_predictor(iter = 500, m = 100, p = 120)
```

```julia
import SemiparametricAUC: simulate_one_predictor

ds_betas = @time simulate_one_predictor(iter = 500, m = 120, p = 100)
@time simulate_one_predictor(iter = 1000, m = 220, p = 200)
@time simulate_one_predictor(iter = 1000, m = 400, p = 300)
@time simulate_one_predictor(iter = 100000, m = 120, p = 100)

meanbeta = DataFrames.colwise(mean, DataFrames.DataFrame(ds_betas[1])) # mean betas
meanvar = DataFrames.colwise(var, DataFrames.DataFrame(ds_betas[2]))  # mean variances
meansd = DataFrames.colwise(sqrt, DataFrames.DataFrame(ds_betas[2]))

#Calculating 95% CI coverage
low_bo  = 0.15 .>= ds_betas[3][:,1]
high_bo = 0.15 .<= ds_betas[3][:,2]
# sum(low_bo & high_bo)/ds_betas[4]

low_b1  = 0.50 .>= ds_betas[3][:,3]
high_b1 = 0.50 .<= ds_betas[3][:,4]
# sum(low_b1 & high_b1)/ds_betas[4]

low_b2  = 1.00 .>= ds_betas[3][:,5]
high_b2 = 1.00 .<= ds_betas[3][:,6]
# sum(low_b2 & high_b2)/ds_betas[4]

all_coverage = vcat(sum(low_bo & high_bo)/ds_betas[4],sum(low_b1 & high_b1)/ds_betas[4],sum(low_b2 & high_b2)/ds_betas[4])
```
