"Calcualtes AUC related estimates"
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

# Example
calculate_auc(ya = DataArray([2,3,2,2]), yb = DataArray([3,2,1,2,.3,4]))
