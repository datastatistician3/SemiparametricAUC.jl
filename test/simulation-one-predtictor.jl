import SemiparametricAUC: simulate_one_predictor

ds_betas = @time simulate_one_predictor(iter = 500, m = 120, p = 100)
@time simulate_one_predictor(iter = 1000, m = 220, p = 200)
@time simulate_one_predictor(iter = 1000, m = 400, p = 300)
@time simulate_one_predictor(iter = 100000, m = 120, p = 100)

meanbeta = DataFrames.colwise(mean, DataFrames.DataFrame(ds_betas[1])) # mean betas
meanvar = DataFrames.colwise(var, DataFrames.DataFrame(ds_betas[2]))  # mean variances
meansd = DataFrames.colwise(sqrt, DataFrames.DataFrame(ds_betas[2]))

#Calculating 95% CI coverage
low_bo = 0.15 .>= ds_betas[3][:,1]
high_bo = 0.15 .<= ds_betas[3][:,2]
# sum(low_bo & high_bo)/ds_betas[4]

low_b1 = 0.50 .>= ds_betas[3][:,3]
high_b1 = 0.50 .<= ds_betas[3][:,4]
# sum(low_b1 & high_b1)/ds_betas[4]

low_b2 = 1.00 .>= ds_betas[3][:,5]
high_b2 = 1.00 .<= ds_betas[3][:,6]
# sum(low_b2 & high_b2)/ds_betas[4]

all_coverage = vcat(sum(low_bo & high_bo)/ds_betas[4],sum(low_b1 & high_b1)/ds_betas[4],sum(low_b2 & high_b2)/ds_betas[4])
