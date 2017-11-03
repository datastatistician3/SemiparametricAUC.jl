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
