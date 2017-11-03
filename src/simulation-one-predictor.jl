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
