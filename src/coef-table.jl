"""
  coefs_table(mf,lo, up, betass, std_errors)

  This function takes a `ModelFrame` object `mf`, numeric arguments `lo`, `up`, `betass` estimates and `std_errors` (beta's
  standard errors, returns a table with model estimates, 95% CI, and p-values.
"""

coefs_table(mf, lo, up, betass, std_errors)

function coefs_table(; mf = throw(ArgumentError("mf is missing")),
   lo = throw(ArgumentError("lo is missing")),
   up = throw(ArgumentError("up is missing")),
   betass = throw(ArgumentError("betass is missing")),
   std_errors = throw(ArgumentError("std_errors is missing")))
  zz = betass ./ std_errors
  result = (StatsBase.CoefTable(hcat(round(betass,4),lo,up,round(std_errors,4),round(zz,4),2.0 * Distributions.ccdf(Distributions.Normal(), abs.(zz))),
             ["Estimate","2.5%","97.5%","Std.Error","t value", "Pr(>|t|)"],
           ["$i" for i = coefnames(mf)], 4))
  return(result)
end
