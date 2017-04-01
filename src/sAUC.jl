
function sAUC(; x::DataFrames.Formula = nothing, treatment_group::Symbol = nothing, data::DataFrames.DataFrame = nothing)
  if (isa(x, Formula))
    input_covariates = DataFrames.Terms(x).terms
    n1 = length(input_covariates)
    input_response = DataFrames.Terms(x).eterms[1]
  else error("Please put response and input as DataFrames.Formula object. For example, x = response ~ x1 + x2")
  end

  if (!isa(treatment_group, Symbol))
    error("The parameter treatment_group should be Symbol object. For e.g. :x1")
  end
  if (!isa(data, DataFrames.DataFrame))
    error("The parameter data should be DataFrames.DataFrame object.")
  end
  input_treatment = Symbol(treatment_group)
  group_covariates = vcat(input_treatment, input_covariates)

  if (sum([isa(data[:,i], DataFrames.PooledDataArray) for i in group_covariates]) == 0)
    error("Please put response and input as formula. For example, response ~ x1 + x2")
  else
      println("Great")
  end

  print_with_color(:red,"Data are being analyzed. Please, be patient.\n\n")
  # split by factors
  grouped_d = DataFrames.groupby(data, group_covariates)

  set1 = Dict()
  for i in 1:Int(0.5*length(grouped_d))
    set1[i] = grouped_d[i]
  end

  set2 = Dict()
  for i in Int(0.5*length(grouped_d)) + 1:length(grouped_d)
    set2[i] = grouped_d[i]
  end

  logitauchat_matrix = collect(calculate_auc(ya = set1[i][:,input_response], yb = set2[i+length(set2)][:,input_response])
    for i in 1:length(set1))

  dff = DataFrames.DataFrame([y[i] for y in logitauchat_matrix, i in 1:length(logitauchat_matrix[1])])
  var_logitauchat = dff[1]
  gamma1 = dff[2]

  # Change to adjust more than 2 d array
  # get levels
  dict_levels = Dict()
  for i in 1:length(input_covariates)
    dict_levels[i] = DataArrays.levels(data[input_covariates[i]])
  end

  # for expand.grid
  matrix_x = collect(Iterators.product(values(sort(dict_levels))...))

  df_from_tuple = DataFrames.DataFrame([y[i] for y in matrix_x, i in 1:length(matrix_x[1])])
  df_from_tuple[input_response] = var_logitauchat

  function convert_to_factor(x)
      return(DataFrames.pool(x))
  end

  for i in input_covariates
    df_from_tuple[i] = convert_to_factor(df_from_tuple[i])
  end

  # model.matrix using DataFrames (ModelMatrix)
  mf = ModelFrame(DataFrames.Terms(x), df_from_tuple)
  mm = ModelMatrix(mf)

  Z = mm.m
  tau  =  diagm([1/i for i in var_logitauchat])
  ztauz = inv(Z' * tau * Z)
  var_betas = diag(ztauz)
  std_error = sqrt(var_betas)
  betas = ztauz * Z' * tau * gamma1

  threshold = Distributions.quantile(Normal(), 0.975)

  lo = betas - threshold*std_error
  up = betas + threshold*std_error
  ci = hcat(betas,lo,up)

  function coeftable(betass = betas, std_errors = std_error)
  zz = betass ./ std_errors
  result = (CoefTable(hcat(round(betass,4),lo,up,round(std_errors,4),round(zz,4),2.0 * ccdf(Normal(), abs.(zz))),
             ["Estimate","2.5%","97.5%","Std.Error","z value", "Pr(>|z|)"],
           ["$i" for i = coefnames(mf)], 4))
  return(result)
  end
  return(coeftable())
end
