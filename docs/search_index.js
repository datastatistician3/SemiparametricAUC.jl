var documenterSearchIndex = {"docs": [

{
    "location": "index.html#",
    "page": "Introduction",
    "title": "Introduction",
    "category": "page",
    "text": ""
},

{
    "location": "index.html#Semi-parametric-Area-Under-the-Curve-(sAUC)-Regression-1",
    "page": "Introduction",
    "title": "Semi-parametric Area Under the Curve (sAUC) Regression",
    "category": "section",
    "text": "Perform AUC analyses with discrete covariates and a semi-parametric estimation"
},

{
    "location": "index.html#What-is-sAUC-model-and-why?-1",
    "page": "Introduction",
    "title": "What is sAUC model and why?",
    "category": "section",
    "text": "In many applications, comparing two groups while adjusting for multiple covariates is desired for the statistical analysis.  For instance, in clinical trials, adjusting for covariates is a necessary aspect of the statistical analysis in order to improve the precision of the treatment comparison and to assess effect modification. sAUC is a semi-parametric AUC regression model to compare the effect of two treatment groups in the intended non-normal outcome while adjusting for discrete covariates. More detailed reasons on what it is and why it is proposed are outlined in this paper. A major reason behind the development of this method is that this method is computationally simple and is based on closed-form parameter and standard error estimation."
},

{
    "location": "index.html#Model-1",
    "page": "Introduction",
    "title": "Model",
    "category": "section",
    "text": "We consider applications that compare a response variable y between two groups (A and B) while adjusting for k categorical covariates X_1X_2X_k.  The response variable y is a continuous or ordinal variable that is not normally distributed.  Without loss of generality, we assume each covariate is coded such that X_i=1n_i,for i=1k. For each combination of the levels of the covariates, we define the Area Under the ROC curve (AUC) in the following way:pi_x_1 x_2x_k=P(Y^AY^BX_1=x_1X_2=x_2X_k=x_k )+frac12 P(Y^A=Y^BX_1=x_1X_2=x_2X_k=x_k )where x_1=1n_1x_k=1n_k, and Y^A and Y^B are two randomly chosen observations from Group A and B, respectively.  The second term in the above equation is for the purpose of accounting ties.For each covariate X_i, without loss of generality, we use the last category as the reference category and define (n_i-1) dummy variables X_i^(1)X_i^(2)X_i^(n_i-1) such thatX_i^(j) (x)= leftbeginarray\nrrr\n1 j = x \n0 j ne x\nendarrayrightwhere i=1k j=1n_i-1 x=1n_i.   We model the association between AUC pi_x_1 x_2x_k and covariates using a logistic model.  Such a model specifies that the logit of pi_x_1 x_2x_k is a linear combination of terms that are products of the dummy variables defined above.  Specifically,logit(pi_x_1 x_2x_k  )=Z_x_1 x_2x_k boldsymbolbetawhere Z_x_1 x_2x_k is a row vector whose elements are zeroes or ones and are products of X_1^(1) (x_1 )X_1^(n_i-1)  (x_1)X_k^(1) (x_k)X_k^(n_k-1) (x_k), and boldsymbolbeta is a column vector of nonrandom unknown parameters.  Now, define a column vector pi by stacking up pi_x_1 x_2x_k and define a matrix Z by stacking up Z_x_1 x_2x_k, as x_i ranges from 1 to n_i i=1k, our final model is  logit(pi)=Zboldsymbolbeta (1)The reason for us to use a logit transformation of the AUC instead of using the original AUC is for variance stabilization.  We will illustrate the above general model using examples."
},

{
    "location": "index.html#Estimation-1",
    "page": "Introduction",
    "title": "Estimation",
    "category": "section",
    "text": "First, we denote the number of observations with covariates X_1=i_1X_k=i_k in groups A and B by N_i_1i_k^A and N_i_1i_k^B, respectively.  We assume both N_i_1i_k^A and N_i_1i_k^B are greater than zero in the following development.  An unbiased estimator of pi_i_1i_k proposed by Mann and Whitney (1947) ishatpi_i_1i_k=fracsum_l=1^N_i_1i_k^A sum_j=1^N_i_1i_k^B I_ljN_i_1i_k^A N_i_1i_k^BwhereI_i_1 i_k lj= leftbeginarray\nrrr\n1 Y_i_1i_k l^AY_i_1i_k j^B \nfrac12 Y_i_1i_k l^A=Y_i_1i_k j^B \n0 Y_i_1i_k l^AY_i_1i_k j^B\nendarrayrightand Y_i_1i_k l^A and Y_i_1i_k j^B are observations with X_1=i_1X_k=i_k in groups A and B, respectively.  Delong, Delong and Clarke-Pearson (1988) have shown thathatpi_i_1i_k approx N(pi_i_1i_ksigma_i_1i_k^2)In order to obtain an estimator for sigma_i_1i_k^2, they first computedV_i_1i_k l^A=frac1N_i_1i_k^B  sum_j=1^N_i_1i_k^B I_lj  	l=1N_i_1i_k^AandV_i_1i_kj^B=frac1N_i_1i_k^A  sum_l=1^N_i_1i_k^A I_lj  	j=1N_i_1i_k^BThen, an estimate of the variance of the nonparametric AUC washatsigma_i_1i_k^2=frac(s_i_1i_k^A )^2N_i_1i_k^A + frac(s_i_1i_k^B )^2N_i_1i_k^Bwhere(s_i_1i_k^A )^2and (s_i_1i_k^B )^2 were the sample variances ofV_i_1i_k l^A l=1N_i_1i_k^Aand V_i_1i_k j^B j=1N_i_1i_k^B respectively.  Clearly, we need both N_i_1i_k^A and N_i_1i_k^B are greater than two in order to compute hatsigma_i_1i_k^2.Now, in order to estimate parameters in Model (1), we first derive the asymptotic variance of hatgamma_i_1i_k using the delta method, which results inhatgamma_i_1i_k=logit(hatpi_i_1i_k) approx N(logit(pi_i_1i_k)tau_i_1i_k^2)where hattau_i_1i_k^2=frachatgamma_i_1i_k^2hatpi_i_1i_k^2  (1-hatpi_i_1i_k)^2Rewriting the above model, we obtainhatgamma_i_1i_k=logit(pi_i_1i_k ) =Z_i_1i_k boldsymbolbeta + epsilon_i_1i_kwhere,epsilon_i_1i_k approx N(0tau_i_1i_k^2).  Then, by stacking up the hatgamma_1_ii_k to be hatgamma Z_i_1i_k to be boldsymbolZ, and epsilon_i_1i_k to be boldsymbolepsilon, we haveboldsymbolhatgamma =logit boldsymbolhatpi = boldsymbolZbeta + epsilonwhere, E(epsilon)=0 and hatT=Var(epsilon)=diag(hattau_i_1 i_k^2) which is a diagonal matrix.  Finally, by using the generalized least squares method, we estimate the parameters beta and its variance-covariance matrix as follows;boldsymbolhatbeta =(hatZ^T  hatT^-1  Z)^-1 Z^T  hatT^-1 hatgammaandThe above equations can be used to construct a 100(1-alpha)% Wald confidence intervals for boldsymbolbeta_i using formulahatbeta_i pm Z_1-fracalpha2 sqrthatV(hatbeta_i)where Z_1-fracalpha2 is the (1-fracalpha2)^th quantile of the standard normal distribution.  Equivalently, we rejectH_0beta_i = 0if hatbeta_i  Z_1-fracalpha2 sqrthatV(hatbeta_i)The p-value for testing H_0 is 2 * P(Z  hatbeta_isqrthatVhatbeta_i)where Z is a random variable with the standard normal distribution.Now, the total number of cells (combinations of covariates X_1X_k is n_1 n_2n_k. As mentioned earlier, for a cell to be usable in the estimation, the cell needs to have at least two observations from Group A and two observations from Group B.  As long as the total number of usable cells is larger than the dimension of boldsymbolbeta, then the matrix boldsymbolhatZ^T  hatT^-1  Z is invertible and consequently,boldsymbolhatbeta is computable and model (1) is identifiable."
},

{
    "location": "install.html#",
    "page": "Installation",
    "title": "Installation",
    "category": "page",
    "text": "InstallationHere is the GitHub repository for SemiparametricAUC.jlYou can install SemiparametricAUC.jl via GitHubgit clone https://github.com/sbohora/SemiparametricAUC.jl.gitThe following installation method is not currently available.Pkg.add(\"SemiparametricAUC\") "
},

{
    "location": "paper.html#",
    "page": "Article",
    "title": "Article",
    "category": "page",
    "text": "Below is the link to our paper published in the Journal of Data Science in 2017.Paper published in the Journal of Data Science"
},

{
    "location": "example-julia.html#",
    "page": "Example",
    "title": "Example",
    "category": "page",
    "text": ""
},

{
    "location": "example-julia.html#Warning:-This-package-is-still-under-development.-1",
    "page": "Example",
    "title": "Warning: This package is still under development.",
    "category": "section",
    "text": ""
},

{
    "location": "example-julia.html#sAUC-in-Julia-(SemiparametricAUC.jl)-1",
    "page": "Example",
    "title": "sAUC in Julia (SemiparametricAUC.jl)",
    "category": "section",
    "text": ""
},

{
    "location": "example-julia.html#Perform-AUC-analyses-with-discrete-covariates-and-a-semi-parametric-estimation-1",
    "page": "Example",
    "title": "Perform AUC analyses with discrete covariates and a semi-parametric estimation",
    "category": "section",
    "text": ""
},

{
    "location": "example-julia.html#Example-1",
    "page": "Example",
    "title": "Example",
    "category": "section",
    "text": "To illustrate how to apply the proposed method, we obtained data from a randomized and controlled clinical trial, which was designed to increase knowledge and awareness to prevent Fetal Alcohol Spectrum Disorders (FASD) in children through the development of printed materials targeting women of childbearing age in Russia. One of the study objectives was to evaluate effects of FASD education brochures with different types of information and visual images on FASD related knowledge, attitudes, and alcohol consumption on childbearing-aged women. The study was conducted in two regions in Russia including St. Petersburg (SPB) and the Nizhny Novgorod Region (NNR) from 2005 to 2008. A total of 458 women were recruited from women's clinics and were randomly assigned to one of three groups (defined by the GROUP variable): (1) a printed FASD prevention brochure with positive images and information stated in a positive way, positive group (PG), (2) a FASD education brochure with negative messages and vivid images, negative group (NG); and (3) a general health material, control group (CG). For the purpose of the analysis in this thesis, only women in the PG and CG were included. Data were obtained from the study principal investigators . The response variable was the change in the number of drinks per day (CHANGE_DRINK=number of drinks after-number of drinks before) on average in the last 30 days from one-month follow-up to baseline. Two covariates considered for the proposed method were \"In the last 30 days, have you smoked cigarettes?\" (SMOKE) and  \"In the last 30 days, did you take any other vitamins?\" (OVITAMIN). Both covariates had \"Yes\" or \"No\" as the two levels. The question of interest here was to assess the joint predictive effects of SMOKE and OVITAMIN on whether the participants reduced the number of drinks per day from baseline to one month follow-up period. A total of 210 women with no missing data on any of the CHANGE_DRINK, SMOKE, GROUP, and OVITAMIN were included in the analysis.The response variable CHANGE_DRINK was heavily skewed and not normally distributed in each group  (Shapiro-Wilk p<0.001). Therefore, we decided to use the AUG regression model to analyze the data.  In the AUG regression model we definelarge pi = p(Y_CG  Y_PG)Note that the value of large pi greater than .5 means that women in the PG had a greater reduction of alcohol drinks than those in the CG. For statistical results, all p-values < .05 were considered statistically significant and 95% CIs were presented.We first fit an AUC regression model including both main effects of the covariates.  Note that the main effects of the covariates in fact represented their interactions with the GROUP variable, which is different than the linear or generalized linear model frame.  The reason is that the GROUP variable is involved in defining the AUC.  Tables below present the parameter estimates, SEs, p-values, and 95% CIs for model with one and two covariates.  Because parameter beta_2 was not significantly different from 0, we dropped OVITAMIN and fit another model including only the SMOKE main effect.Table below shows a significant interaction between SMOKE and GROUP because the SMOKE was statistically significant (95% CI: (0.05, 1.47)). Therefore, the final model was logit(hatpi_Smoke) = hatbeta_0 + hatbeta_1*I(Smoke =Yes)Because the interaction between SMOKE and GROUP was significant, we need to use AUC as a measure of the GROUP effect on CHANGE_DRINK for smokers and non-smokers separately using following formula for example for smokers;hatpi_Smoke = frace^hatbeta_0 + hatbeta_1*Smoke =Yes1 + e^hatbeta_0 + hatbeta_1*Smoke =YesSpecifically, the AUCs were 0.537 (insignificant) and 0.713 (significant) for non-smokers and smokers, respectively.  This implies that the effect of positive and control brochures were similar for nonsmokers; however, for smokers, the probability that the positive brochure had a better effect than the control brochure in terms of alcohol reduction is 71.30%, indicating the positive brochure is a better option than the control brochure."
},

{
    "location": "example-julia.html#Result-of-sAUC-Regression-with-one-discrete-covariate-1",
    "page": "Example",
    "title": "Result of sAUC Regression with one discrete covariate",
    "category": "section",
    "text": "using DataFrames\nusing SemiparametricAUC\n\n# Data analysis examples\nfasd = DataFrames.readtable(joinpath(Pkg.dir(\"SemiparametricAUC\"), \"data/fasd.csv\"))\n# fasd = readtable(\"ds.csv\")\n\n# Define factor/categorical variable\nfunction convert_to_factor(x)\n    return(DataFrames.pool(x))\nend\n\nfasd[:group] = convert_to_factor(fasd[:group])\nfasd[:x1]    = convert_to_factor(fasd[:x1])\nfasd[:x2]    = convert_to_factor(fasd[:x2])\n# fasd[:x3]  = convert_to_factor(fasd[:x3])\n\none_covariates_results = SemiparametricAUC.semiparametricAUC(model_formula = y ~ x1, treatment_group = :group, data = fasd)\none_covariates_results"
},

{
    "location": "example-julia.html#Model-Summary:-one-discrete-covariate-1",
    "page": "Example",
    "title": "Model Summary: one discrete covariate",
    "category": "section",
    "text": "Predictors Estimate 2.5% 97.5% Std. Error t p\n(Intercept) -0.1432 -0.471359 0.185059 0.1675 -0.8548 0.392634\nx1: 1 -0.7668 -1.47803 -0.0555374 0.3629 -2.113 0.0346002"
},

{
    "location": "example-julia.html#Result-of-sAUC-Regression-with-two-discrete-covariates-1",
    "page": "Example",
    "title": "Result of sAUC Regression with two discrete covariates",
    "category": "section",
    "text": "two_covariates_results = SemiparametricAUC.semiparametricAUC(model_formula = y ~ x1 + x2, treatment_group = :group, data = fasd)\ntwo_covariates_results"
},

{
    "location": "example-julia.html#Model-Summary:-two-discrete-covariates-1",
    "page": "Example",
    "title": "Model Summary: two discrete covariates",
    "category": "section",
    "text": "Predictors Estimate 2.5% 97.5% Std. Error t p\n(Intercept) -0.1034 -0.49026 0.283465 0.1974 -0.5238 0.600387\nx1: 1 -0.2189 -0.881207 0.44348 0.3379 -0.6476 0.517213\nx2: 1 -0.7434 -1.46562 -0.021217 0.3685 -2.0175 0.0436388"
},

{
    "location": "command.html#",
    "page": "References",
    "title": "References",
    "category": "page",
    "text": "CurrentModule = SemiparametricAUC"
},

{
    "location": "command.html#SemiparametricAUC.calculate_auc-Tuple{Any,Any}",
    "page": "References",
    "title": "SemiparametricAUC.calculate_auc",
    "category": "Method",
    "text": "calculate_auc(ya, yb)\n\n\"Calcualtes AUC related estimates\"\n\nThis function takes two DataArray arguments ya and yb and calculates variance of predicted AUC,   logit of predicted AUC, and variance of logit of predicted AUC responses passed.\n\n\n\n"
},

{
    "location": "command.html#SemiparametricAUC.semiparametricAUC-Tuple{Any,Any,Any}",
    "page": "References",
    "title": "SemiparametricAUC.semiparametricAUC",
    "category": "Method",
    "text": "semiparametricAUC(model_formula, treatment_group, data)\n\nThis function is used to fit semiparametric AUC regression model specified by   giving a formula object of response and covariates and a separate argument of treatment   group. It will convert variables other than response into factors, estimate model parameters,   and display results.\n\nThis function takes a formula object model_formula with response and covariates such as {response ~ x1 + x2},   group argument treatment_group, which is treatment group for which a comparision is to be made, and    a data argument data which is a DataFrame object that contains variables needed for the analysis.\n\n\n\n"
},

{
    "location": "command.html#SemiparametricAUC.calculate_auc_simulation-Tuple{Any,Any}",
    "page": "References",
    "title": "SemiparametricAUC.calculate_auc_simulation",
    "category": "Method",
    "text": "calculate_auc_simulation(ya, yb)\n\nThis function takes two DataArray arguments ya and yb and calculates variance of predicted AUC,   logit of predicted AUC, and variance of logit of predicted AUC responses passed.\n\n\n\n"
},

{
    "location": "command.html#SemiparametricAUC.simulate_one_predictor",
    "page": "References",
    "title": "SemiparametricAUC.simulate_one_predictor",
    "category": "Function",
    "text": "simulate_one_predictor(iter, m, p)\n\nIt asks for number of iterations iter to be run, number of observations m in treatment   and control groups p for the simulation of Semiparametric AUC regression adjusting for one discrete   covariate. In this simulation, true model parameters are as follows: β0 = 0.15, β1 = 0.50, β2 = 1.\n\n\n\n"
},

{
    "location": "command.html#SemiparametricAUC.coefs_table-Tuple{Any,Any,Any,Any,Any}",
    "page": "References",
    "title": "SemiparametricAUC.coefs_table",
    "category": "Method",
    "text": "coef_table(mf,lo, up, betass, std_errors)\n\nThis function takes a ModelFrame object mf, numeric arguments lo, up, betass estimates and std_errors (beta's   standard errors, returns a table with model estimates, 95% CI, and p-values.\n\n\n\n"
},

{
    "location": "command.html#SemiparametricAUC.jl-Commands-1",
    "page": "References",
    "title": "SemiparametricAUC.jl Commands",
    "category": "section",
    "text": "calculate_auc(ya, yb)semiparametricAUC(model_formula, treatment_group, data)calculate_auc_simulation(ya, yb)simulate_one_predictor(iter = 500, m = 100, p = 120)coefs_table(mf, lo, up, betass, std_errors)"
},

{
    "location": "code.html#",
    "page": "Code",
    "title": "Code",
    "category": "page",
    "text": ""
},

{
    "location": "code.html#SemiparametricAUC.jl-Code-1",
    "page": "Code",
    "title": "SemiparametricAUC.jl Code",
    "category": "section",
    "text": ""
},

{
    "location": "code.html#calculate_auc-1",
    "page": "Code",
    "title": "calculate_auc",
    "category": "section",
    "text": "\"\"\"\n  calculate_auc(ya, yb)\n\n  # \"Calcualtes AUC related estimates\"\n\n  This function takes two DataArray arguments `ya` and `yb` and calculates variance of predicted AUC,\n  logit of predicted AUC, and variance of logit of predicted AUC responses passed.\n\"\"\"\ncalculate_auc(ya = DataFrames.DataArray([2,3,4,3]), yb = DataFrames.DataArray([3,2,1,3,4,3]))\n\nfunction calculate_auc(; ya::DataArrays.DataArray = nothing, yb::DataArrays.DataArray = nothing)\n  m = length(ya)\n  p = length(yb)\n  I = zeros(m, p)\n    for i in range(1,m)\n        for j in range(1,p)\n            if ya[i] > yb[j]\n              I[i,j] = 1\n            elseif ya[i] == yb[j]\n              I[i,j] = 0.5\n            else\n               I[i,j] = 0\n            end\n        end\n    end\n    finv(x::Float64) = return(-log((1/x)-1))\n    auchat = mean(I)\n    finvhat = finv(auchat)\n    vya = mean(I,2)\n    vyb = mean(I,1)\n    svarya = var(vya)\n    svaryb = var(vyb)\n    vhat_auchat = (svarya/m) + (svaryb/p)\n    v_finv_auchat = vhat_auchat/((auchat^2)*(1-auchat)^2)\n    logitauchat = log(auchat/(1-auchat))\n    var_logitauchat = vhat_auchat /((auchat^2)*(1-auchat)^2)\n    return((var_logitauchat, logitauchat))\nend"
},

{
    "location": "code.html#Example-1",
    "page": "Code",
    "title": "Example",
    "category": "section",
    "text": "calculate_auc(ya = DataFrames.DataArray([2,3,4,5]), yb = DataFrames.DataArray([2,3,4,5]))"
},

{
    "location": "code.html#semiparametricAUC-1",
    "page": "Code",
    "title": "semiparametricAUC",
    "category": "section",
    "text": "\"\"\"\n  semiparametricAUC(model_formula, treatment_group, data)\n\n  This function is used to fit semiparametric AUC regression model specified by\n  giving a formula object of response and covariates and a separate argument of treatment\n  group. It will convert variables other than response into factors, estimate model parameters,\n  and display results.\n\n  This function takes a formula object `model_formula` with response and covariates such as {response ~ x1 + x2},\n  group argument `treatment_group`, which is treatment group for which a comparision is to be made, and\n  a data argument `data` which is a DataFrame object that contains variables needed for the analysis.\n\"\"\"\nsemiparametricAUC(model_formula = y ~ x1, treatment_group = :group, data = fasd)\n\nfunction semiparametricAUC(; model_formula::DataFrames.Formula = throw(ArgumentError(\"Argument model_formula is missing\")),\n  treatment_group::Symbol = throw(ArgumentError(\"Argument treatment_group is missing\")),\n    data::DataFrames.DataFrame = throw(ArgumentError(\"Argument data is missing\")))\n  # fasd = DataFrames.readtable(joinpath(Pkg.dir(\"SemiparametricAUC\"), \"data/fasd.csv\"))\n  # model_formula = y ~ x1 + x2\n  # treatment_group = :group\n  # data = fasd\n\n  if (isa(model_formula, Formula))\n    input_covariates = DataFrames.Terms(model_formula).terms\n    n1 = length(input_covariates)\n    input_response = Symbol(DataFrames.Terms(model_formula).eterms[1])\n  else error(\"Please put response and input as DataFrames.Formula object. For example, model_formula = response ~ x1 + x2\")\n  end\n\n  if (!isa(treatment_group, Symbol))\n    error(\"The parameter treatment_group should be Symbol object. For e.g. :x1\")\n  end\n\n  if (!isa(data, DataFrames.DataFrame))\n    error(\"The parameter data should be DataFrames.DataFrame object.\")\n  end\n  input_treatment = Symbol(treatment_group)\n  group_covariates = vcat(input_covariates, input_treatment)\n\n  # if (sum([isa(data[:,i], DataFrames.PooledDataArray) for i in group_covariates]) == 0)\n  #   error(\"Please put response and input as formula. For example, response ~ x1 + x2\")\n  # else\n  #     println(\"Great\")\n  # end\n\n  print_with_color(:red,\"Data are being analyzed. Please, be patient.\\n\\n\")\n  # split by factors\n  # TO-DO: make sure that oerder of the variables are aligned with order of coefnames(mf)\n  ds = data[vcat(input_response,Symbol.(group_covariates))]\n  grouped_d = DataFrames.groupby(ds, group_covariates)\n\n  half_data = 0.5\n  set1 =  Dict()\n  for i in 1:2:length(grouped_d)\n    set1[i] = grouped_d[i]\n  end\n  set1_sorted = DataStructures.SortedDict(set1)\n\n  set2 = Dict()\n  for i in 2:2:length(grouped_d)\n    set2[i] = grouped_d[i]\n  end\n  set2_sorted = DataStructures.SortedDict(set2)\n\n  # TO-DO: make sure that order of the variables are aligned with order of coefnames(mf)\n  logitauchat_matrix = collect(calculate_auc(ya = set1_sorted[i][:,input_response], yb = set2_sorted[i+1][:,input_response])\n    for i in 1:2:length(grouped_d))\n\n  dff = DataFrames.DataFrame([y[i] for y in logitauchat_matrix, i in 1:length(logitauchat_matrix[1])])\n  var_logitauchat = dff[1]\n  gamma1 = dff[2]\n\n  # Change to adjust more than 2 d array\n  # get levels\n  dict_levels = Dict()\n  for i in 1:length(input_covariates)\n    dict_levels[i] = DataArrays.levels(ds[input_covariates[i]])\n  end\n\n  # for expand.grid\n  matrix_x = collect(Iterators.product(values(sort(dict_levels))...))\n\n  df_from_tuple = DataFrames.DataFrame([y[i] for y in matrix_x, i in 1:length(matrix_x[1])])\n  df_from_tuple[input_response] = gamma1\n  df_from_tuple[:var_logit] = var_logitauchat\n\n  function convert_to_factor(x)\n      return(DataFrames.pool(x))\n  end\n\n  for i in input_covariates\n    df_from_tuple[i] = convert_to_factor(df_from_tuple[i])\n  end\n\n  # model.matrix using DataFrames (ModelMatrix)\n  mf = ModelFrame(DataFrames.Terms(model_formula), df_from_tuple)\n  mm = ModelMatrix(mf)\n\n  coefnames(mf)\n  Z = mm.m\n  tau  =  diagm([1/i for i in var_logitauchat])\n  ztauz = inv(Z' * tau * Z)\n  var_betas = diag(ztauz)\n  std_error = sqrt(var_betas)\n  betas = ztauz * Z' * tau * gamma1\n\n  threshold = Distributions.quantile(Distributions.Normal(), 0.975)\n\n  lo = betas - threshold*std_error\n  up = betas + threshold*std_error\n  ci = hcat(betas,lo,up)\n\n  return(SemiparametricAUC.coefs_table(mf = mf, lo = lo, up = up, betass = betas, std_errors = std_error))\nend"
},

{
    "location": "code.html#Example-2",
    "page": "Code",
    "title": "Example",
    "category": "section",
    "text": "semiparametricAUC(model_formula = y ~ x1, treatment_group = :group, data = fasd)\"\"\"\n  coef_table(mf,lo, up, betass, std_errors)\n\n  This function takes a `ModelFrame` object `mf`, numeric arguments `lo`, `up`, `betass` estimates and `std_errors` (beta's\n  standard errors, returns a table with model estimates, 95% CI, and p-values.\n\"\"\"\n\ncoefs_table(mf, lo, up, betass, std_errors)\n\nfunction coefs_table(; mf = throw(ArgumentError(\"mf is missing\")),\n   lo = throw(ArgumentError(\"lo is missing\")),\n   up = throw(ArgumentError(\"up is missing\")),\n   betass = throw(ArgumentError(\"betass is missing\")),\n   std_errors = throw(ArgumentError(\"std_errors is missing\")))\n  zz = betass ./ std_errors\n  result = (StatsBase.CoefTable(hcat(round(betass,4),lo,up,round(std_errors,4),round(zz,4),2.0 * Distributions.ccdf(Distributions.Normal(), abs.(zz))),\n             [\"Estimate\",\"2.5%\",\"97.5%\",\"Std.Error\",\"t value\", \"Pr(>|t|)\"],\n           [\"$i\" for i = coefnames(mf)], 4))\n  return(result)\nendjulia"
},

{
    "location": "code.html#Example-3",
    "page": "Code",
    "title": "Example",
    "category": "section",
    "text": "coef_table(mf,lo, up, betass, std_errors)"
},

{
    "location": "code.html#calculate_auc_simulation-1",
    "page": "Code",
    "title": "calculate_auc_simulation",
    "category": "section",
    "text": "\"\"\"\n  calculate_auc_simulation(ya, yb)\n\n  This function takes two DataArray arguments `ya` and `yb` and calculates variance of predicted AUC,\n  logit of predicted AUC, and variance of logit of predicted AUC responses passed.\n\"\"\"\n\ncalculate_auc_simulation(ya = DataFrames.DataArray([2,3,4,3]), yb = DataFrames.DataArray([3,2,1,3,4,3]))\n\nfunction calculate_auc_simulation(; ya::Array = nothing, yb::Array = nothing)\n  m = length(ya)\n  p = length(yb)\n  I = zeros(m, p)\n    for i in range(1,m)\n        for j in range(1,p)\n            if ya[i] > yb[j]\n              I[i,j] = 1\n            elseif ya[i] == yb[j]\n              I[i,j] = 0.5\n            else\n               I[i,j] = 0\n            end\n        end\n    end\n    finv(x::Float64) = return(-log((1/x)-1))\n    auchat = mean(I)\n    finvhat = finv(auchat)\n    vya = mean(I,2)\n    vyb = mean(I,1)\n    svarya = var(vya)\n    svaryb = var(vyb)\n    vhat_auchat = (svarya/m) + (svaryb/p)\n    v_finv_auchat = vhat_auchat/((auchat^2)*(1-auchat)^2)\n    logitauchat = log(auchat/(1-auchat))\n    var_logitauchat = vhat_auchat /((auchat^2)*(1-auchat)^2)\n    return(auchat, finvhat, vhat_auchat)\nend"
},

{
    "location": "code.html#Example-4",
    "page": "Code",
    "title": "Example",
    "category": "section",
    "text": "calculate_auc_simulation(ya = DataFrames.DataArray([2,3,4,3]), yb = DataFrames.DataArray([3,2,1,3,4,3]))"
},

{
    "location": "code.html#simulate_one_predictor-1",
    "page": "Code",
    "title": "simulate_one_predictor",
    "category": "section",
    "text": "\"\"\"\n  simulate_one_predictor(iter, m, p)\n\n  It asks for number of iterations `iter` to be run, number of observations `m` in treatment\n  and control groups `p` for the simulation of Semiparametric AUC regression adjusting for one discrete\n  covariate. In this simulation, true model parameters are as follows: β0 = 0.15, β1 = 0.50, β2 = 1.\n\"\"\"\nsimulate_one_predictor(iter = 500, m = 100, p = 120)\n\nfunction simulate_one_predictor(;iter = 500, m = 100, p = 120)\n    iter = iter\n    finvhat = gamma1 = Array(Float64, 3)\n    AUChat = Array(Float64, 3)\n    Vhat_auchat = Array(Float64, 3)\n    lo = up = Array(Float64, iter, 3)\n    d  = Array(Float64, m, 3)\n    nd = Array(Float64, p, 3)\n\n    m_betas = Array(Float64, iter, 3)\n    sd_betas = Array(Float64, iter, 3)\n    lower = upper = score =  cov_b = Array(Float64, iter, 3)\n    v_finv_auchat = gamma = Array(Float64, 3)\n    all_var = Array(Float64, 3)\n    var_finv_auchat = Array(Float64, iter, 3)\n\n    for z in range(1,iter)\n      d1 = [0.0, 0.0, 0.0]\n      d2 = [0, 0.50, 1.00]\n      d0 = 0.15\n      y = 1:3\n      for k in y\n        result = Array(Float64, k, 3)\n        u1 = randexp(p) # rand(Exponential(1), p)\n        u2 = randexp(m) # rand(Exponential(1), m)\n        d[:,k]=-log(u2) + d0 +(d1[k] + d2[k])\n        nd[:,k]=-log(u1) + d1[k]\n\n        result= calculate_auc_simulation(ya = d[:,k], yb = nd[:,k])\n        AUChat[k]=result[1]\n        finvhat[k]=result[2]\n        Vhat_auchat[k] = result[3]\n        v_finv_auchat[k] = Vhat_auchat[k]/(((AUChat[k])^2)*(1-(AUChat[k]))^2)  #Variance of F inverse\n      end\n      gamma1 = finvhat\n      Z = reshape([1,0,0,1,1,0,1,0,1], 3, 3)'\n\n      tau  =  diagm([1/i for i in v_finv_auchat])\n\n      ztauz = inv(Z' * tau * Z)\n      var_betas = diag(ztauz)\n      std_error = sqrt(var_betas)\n      betas = ztauz * Z' * tau * gamma1\n\n      m_betas[z,:]  =  betas\n      var_finv_auchat[z,:] = var_betas\n    end\n    lo = m_betas .- 1.96*sqrt(var_finv_auchat)\n    up = m_betas .+ 1.96*sqrt(var_finv_auchat)\n    ci = hcat(lo,up)\n    ci_betas = ci[:,[1,4,2,5,3,6]]\n    return(m_betas, var_finv_auchat, ci_betas, iter)\nend"
},

{
    "location": "code.html#Example-5",
    "page": "Code",
    "title": "Example",
    "category": "section",
    "text": "simulate_one_predictor(iter = 500, m = 100, p = 120)import SemiparametricAUC: simulate_one_predictor\n\nds_betas = @time simulate_one_predictor(iter = 500, m = 120, p = 100)\n@time simulate_one_predictor(iter = 1000, m = 220, p = 200)\n@time simulate_one_predictor(iter = 1000, m = 400, p = 300)\n@time simulate_one_predictor(iter = 100000, m = 120, p = 100)\n\nmeanbeta = DataFrames.colwise(mean, DataFrames.DataFrame(ds_betas[1])) # mean betas\nmeanvar = DataFrames.colwise(var, DataFrames.DataFrame(ds_betas[2]))  # mean variances\nmeansd = DataFrames.colwise(sqrt, DataFrames.DataFrame(ds_betas[2]))\n\n#Calculating 95% CI coverage\nlow_bo  = 0.15 .>= ds_betas[3][:,1]\nhigh_bo = 0.15 .<= ds_betas[3][:,2]\n# sum(low_bo & high_bo)/ds_betas[4]\n\nlow_b1  = 0.50 .>= ds_betas[3][:,3]\nhigh_b1 = 0.50 .<= ds_betas[3][:,4]\n# sum(low_b1 & high_b1)/ds_betas[4]\n\nlow_b2  = 1.00 .>= ds_betas[3][:,5]\nhigh_b2 = 1.00 .<= ds_betas[3][:,6]\n# sum(low_b2 & high_b2)/ds_betas[4]\n\nall_coverage = vcat(sum(low_bo & high_bo)/ds_betas[4],sum(low_b1 & high_b1)/ds_betas[4],sum(low_b2 & high_b2)/ds_betas[4])"
},

{
    "location": "example.html#",
    "page": "sAUC in R",
    "title": "sAUC in R",
    "category": "page",
    "text": "Please follow this link to R package site sAUC in R (sAUC)"
},

{
    "location": "example.html#Perform-AUC-analyses-with-discrete-covariates-and-a-semi-parametric-estimation-1",
    "page": "sAUC in R",
    "title": "Perform AUC analyses with discrete covariates and a semi-parametric estimation",
    "category": "section",
    "text": ""
},

{
    "location": "example.html#Installation-1",
    "page": "sAUC in R",
    "title": "Installation",
    "category": "section",
    "text": "devtools::install_github(\"sbohora/sAUC\")"
},

{
    "location": "example.html#Example-1",
    "page": "sAUC in R",
    "title": "Example",
    "category": "section",
    "text": "To illustrate how to apply the proposed method, we obtained data from a randomized and controlled clinical trial, which was designed to increase knowledge and awareness to prevent Fetal Alcohol Spectrum Disorders (FASD) in children through the development of printed materials targeting women of childbearing age in Russia. One of the study objectives was to evaluate effects of FASD education brochures with different types of information and visual images on FASD related knowledge, attitudes, and alcohol consumption on childbearing-aged women. The study was conducted in two regions in Russia including St. Petersburg (SPB) and the Nizhny Novgorod Region (NNR) from 2005 to 2008. A total of 458 women were recruited from women's clinics and were randomly assigned to one of three groups (defined by the GROUP variable): (1) a printed FASD prevention brochure with positive images and information stated in a positive way, positive group (PG), (2) a FASD education brochure with negative messages and vivid images, negative group (NG); and (3) a general health material, control group (CG). For the purpose of the analysis in this thesis, only women in the PG and CG were included. Data were obtained from the study principal investigators . The response variable was the change in the number of drinks per day (CHANGE_DRINK=number of drinks after-number of drinks before) on average in the last 30 days from one-month follow-up to baseline. Two covariates considered for the proposed method were \"In the last 30 days, have you smoked cigarettes?\" (SMOKE) and  \"In the last 30 days, did you take any other vitamins?\" (OVITAMIN). Both covariates had \"Yes\" or \"No\" as the two levels. The question of interest here was to assess the joint predictive effects of SMOKE and OVITAMIN on whether the participants reduced the number of drinks per day from baseline to one month follow-up period. A total of 210 women with no missing data on any of the CHANGE_DRINK, SMOKE, GROUP, and OVITAMIN were included in the analysis.The response variable CHANGE_DRINK was heavily skewed and not normally distributed in each group  (Shapiro-Wilk p<0.001). Therefore, we decided to use the AUG regression model to analyze the data.  In the AUG regression model we definelarge pi = p(Y_CG  Y_PG)Note that the value of large pi greater than .5 means that women in the PG had a greater reduction of alcohol drinks than those in the CG. For statistical results, all p-values < .05 were considered statistically significant and 95% CIs were presented.We first fit an AUC regression model including both main effects of the covariates.  Note that the main effects of the covariates in fact represented their interactions with the GROUP variable, which is different than the linear or generalized linear model frame.  The reason is that the GROUP variable is involved in defining the AUC.  Tables below present the parameter estimates, SEs, p-values, and 95% CIs for model with one and two covariates.  Because parameter beta_2 was not significantly different from 0, we dropped OVITAMIN and fit another model including only the SMOKE main effect.Table below shows a significant interaction between SMOKE and GROUP because the SMOKE was statistically significant (95% CI: (0.05, 1.47)). Therefore, the final model was logit(hatpi_Smoke) = hatbeta_0 + hatbeta_1*I(Smoke =Yes)Because the interaction between SMOKE and GROUP was significant, we need to use AUC as a measure of the GROUP effect on CHANGE_DRINK for smokers and non-smokers separately using following formula for example for smokers;hatpi_Smoke = frace^hatbeta_0 + hatbeta_1*Smoke =Yes1 + e^hatbeta_0 + hatbeta_1*Smoke =YesSpecifically, the AUCs were 0.537 (insignificant) and 0.713 (significant) for non-smokers and smokers, respectively.  This implies that the effect of positive and control brochures were similar for nonsmokers; however, for smokers, the probability that the positive brochure had a better effect than the control brochure in terms of alcohol reduction is 71.30%, indicating the positive brochure is a better option than the control brochure."
},

{
    "location": "example.html#Result-of-sAUC-Regression-with-one-discrete-covariate-1",
    "page": "sAUC in R",
    "title": "Result of sAUC Regression with one discrete covariate",
    "category": "section",
    "text": "library(sAUC)\nlibrary(DT)\nlibrary(shiny)\n\nfasd_label <- sAUC::fasd\nnames(fasd_label) <- c(\"y\", \"group\", \"vitamin\", \"smoke\")\nfasd_label[, c(\"smoke\", \"vitamin\", \"group\")] <- lapply(fasd_label[, c(\"smoke\", \"vitamin\", \"group\")], function(x) factor(x))\n\nresult_one <- sAUC::sAUC(formula = y ~ smoke, treatment_group = \"group\", data = fasd_label)The model is:  logit [ p ( Y_1  >  Y_2 ) ]  =  beta_0 +  beta_1*smoke2 \n\nModel Summaryresult_two <- sAUC::sAUC(formula = y ~ smoke + vitamin, treatment_group = \"group\", data = fasd_label)The model is:  logit [ p ( Y_1  >  Y_2 ) ]  =  beta_0 +  beta_1*smoke2 + beta_2*vitamin2  \n\nModel Summaryresult_one$`Model summary`\n            Coefficients Std. Error    2.5%   97.5% Pr(>|z|)\n(Intercept)      -0.9099     0.3219 -1.5409 -0.2789   0.0047\nsmoke2            0.7668     0.3629  0.0555  1.4780   0.0346\n\n$Coefficients\n                  [,1]\n(Intercept) -0.9099357\nsmoke2       0.7667856\n\n$`AUC details`\n     auchat    finvhat logitauchat v_finv_auchat var_logitauchat\n1 0.2870130 -0.9099357  -0.9099357    0.10364630      0.10364630\n2 0.4642734 -0.1431502  -0.1431502    0.02804169      0.02804169\n\n$`Session information`\nR version 3.4.2 (2017-09-28)\nPlatform: x86_64-w64-mingw32/x64 (64-bit)\nRunning under: Windows 7 x64 (build 7601) Service Pack 1\n\nMatrix products: default\n\nlocale:\n[1] LC_COLLATE=English_United States.1252 \n[2] LC_CTYPE=English_United States.1252   \n[3] LC_MONETARY=English_United States.1252\n[4] LC_NUMERIC=C                          \n[5] LC_TIME=English_United States.1252    \n\nattached base packages:\n[1] stats     graphics  grDevices utils     datasets  methods   base     \n\nother attached packages:\n[1] shiny_1.0.5  DT_0.2       sAUC_0.0.1.9\n\nloaded via a namespace (and not attached):\n [1] Rcpp_0.12.13    digest_0.6.12   rprojroot_1.2   mime_0.5       \n [5] R6_2.2.2        xtable_1.8-2    backports_1.1.1 magrittr_1.5   \n [9] evaluate_0.10.1 stringi_1.1.5   rmarkdown_1.6   tools_3.4.2    \n[13] stringr_1.2.0   htmlwidgets_0.9 httpuv_1.3.5    yaml_2.1.14    \n[17] compiler_3.4.2  htmltools_0.3.6 knitr_1.17     \n\n$`Matrix of unique X levels `\n  smoke\n1     1\n2     2\n\n$`Design matrix`\n  (Intercept) smoke2\n1           1      0\n2           1      1\nattr(,\"assign\")\n[1] 0 1\nattr(,\"contrasts\")\nattr(,\"contrasts\")$smoke\n[1] \"contr.treatment\"\n\n\n$model_formula\n[1] \"logit [ p ( Y 1  >  Y 2 ) ] \\n\\n\"\n\n$input_covariates\n[1] \"smoke\"\n\n$input_response\n[1] \"y\""
},

{
    "location": "example.html#Result-of-sAUC-Regression-with-two-discrete-covariates-1",
    "page": "sAUC in R",
    "title": "Result of sAUC Regression with two discrete covariates",
    "category": "section",
    "text": "result_two$`Model summary`\n            Coefficients Std. Error    2.5%   97.5% Pr(>|z|)\n(Intercept)      -1.0657     0.4326 -1.9136 -0.2177   0.0138\nsmoke2            0.7434     0.3685  0.0212  1.4656   0.0436\nvitamin2          0.2189     0.3379 -0.4435  0.8812   0.5172\n\n$Coefficients\n                  [,1]\n(Intercept) -1.0656809\nsmoke2       0.7434195\nvitamin2     0.2188638\n\n$`AUC details`\n     auchat    finvhat logitauchat v_finv_auchat var_logitauchat\n1 0.0937500 -2.2686835  -2.2686835    0.69577034      0.69577034\n2 0.4592934 -0.1631876  -0.1631876    0.09200219      0.09200219\n3 0.3467078 -0.6335421  -0.6335421    0.12335006      0.12335006\n4 0.4566667 -0.1737693  -0.1737693    0.04070024      0.04070024\n\n$`Session information`\nR version 3.4.2 (2017-09-28)\nPlatform: x86_64-w64-mingw32/x64 (64-bit)\nRunning under: Windows 7 x64 (build 7601) Service Pack 1\n\nMatrix products: default\n\nlocale:\n[1] LC_COLLATE=English_United States.1252 \n[2] LC_CTYPE=English_United States.1252   \n[3] LC_MONETARY=English_United States.1252\n[4] LC_NUMERIC=C                          \n[5] LC_TIME=English_United States.1252    \n\nattached base packages:\n[1] stats     graphics  grDevices utils     datasets  methods   base     \n\nother attached packages:\n[1] shiny_1.0.5  DT_0.2       sAUC_0.0.1.9\n\nloaded via a namespace (and not attached):\n [1] Rcpp_0.12.13    digest_0.6.12   rprojroot_1.2   mime_0.5       \n [5] R6_2.2.2        xtable_1.8-2    backports_1.1.1 magrittr_1.5   \n [9] evaluate_0.10.1 stringi_1.1.5   rmarkdown_1.6   tools_3.4.2    \n[13] stringr_1.2.0   htmlwidgets_0.9 httpuv_1.3.5    yaml_2.1.14    \n[17] compiler_3.4.2  htmltools_0.3.6 knitr_1.17     \n\n$`Matrix of unique X levels `\n  smoke vitamin\n1     1       1\n2     2       1\n3     1       2\n4     2       2\n\n$`Design matrix`\n  (Intercept) smoke2 vitamin2\n1           1      0        0\n2           1      1        0\n3           1      0        1\n4           1      1        1\nattr(,\"assign\")\n[1] 0 1 2\nattr(,\"contrasts\")\nattr(,\"contrasts\")$smoke\n[1] \"contr.treatment\"\n\nattr(,\"contrasts\")$vitamin\n[1] \"contr.treatment\"\n\n\n$model_formula\n[1] \"logit [ p ( Y 1  >  Y 2 ) ] \\n\\n\"\n\n$input_covariates\n[1] \"smoke\"   \"vitamin\"\n\n$input_response\n[1] \"y\""
},

{
    "location": "r-shiny.html#",
    "page": "sAUC in R Shiny",
    "title": "sAUC in R Shiny",
    "category": "page",
    "text": "Below is the link to R Shiny application for the proposed method deployed using shinyapps.io by Rstudio .R Shiny Application"
},

{
    "location": "example-python.html#",
    "page": "saucpy in Python",
    "title": "saucpy in Python",
    "category": "page",
    "text": "Please follow this link to Python package site sAUC in Python"
},

{
    "location": "example-python.html#Perform-AUC-analyses-with-discrete-covariates-and-a-semi-parametric-estimation-1",
    "page": "saucpy in Python",
    "title": "Perform AUC analyses with discrete covariates and a semi-parametric estimation",
    "category": "section",
    "text": ""
},

{
    "location": "example-python.html#Installation-1",
    "page": "saucpy in Python",
    "title": "Installation",
    "category": "section",
    "text": "You can install saucpy via GitHub$ git clone https://github.com/sbohora/saucpy.gitThe following installation method is not currently available.$ pip install saucpy"
},

{
    "location": "example-python.html#Example-1",
    "page": "saucpy in Python",
    "title": "Example",
    "category": "section",
    "text": "To illustrate how to apply the proposed method, we obtained data from a randomized and controlled clinical trial, which was designed to increase knowledge and awareness to prevent Fetal Alcohol Spectrum Disorders (FASD) in children through the development of printed materials targeting women of childbearing age in Russia. One of the study objectives was to evaluate effects of FASD education brochures with different types of information and visual images on FASD related knowledge, attitudes, and alcohol consumption on childbearing-aged women. The study was conducted in two regions in Russia including St. Petersburg (SPB) and the Nizhny Novgorod Region (NNR) from 2005 to 2008. A total of 458 women were recruited from women's clinics and were randomly assigned to one of three groups (defined by the GROUP variable): (1) a printed FASD prevention brochure with positive images and information stated in a positive way, positive group (PG), (2) a FASD education brochure with negative messages and vivid images, negative group (NG); and (3) a general health material, control group (CG). For the purpose of the analysis in this thesis, only women in the PG and CG were included. Data were obtained from the study principal investigators . The response variable was the change in the number of drinks per day (CHANGE_DRINK=number of drinks after-number of drinks before) on average in the last 30 days from one-month follow-up to baseline. Two covariates considered for the proposed method were \"In the last 30 days, have you smoked cigarettes?\" (SMOKE) and  \"In the last 30 days, did you take any other vitamins?\" (OVITAMIN). Both covariates had \"Yes\" or \"No\" as the two levels. The question of interest here was to assess the joint predictive effects of SMOKE and OVITAMIN on whether the participants reduced the number of drinks per day from baseline to one month follow-up period. A total of 210 women with no missing data on any of the CHANGE_DRINK, SMOKE, GROUP, and OVITAMIN were included in the analysis.The response variable CHANGE_DRINK was heavily skewed and not normally distributed in each group  (Shapiro-Wilk p<0.001). Therefore, we decided to use the AUG regression model to analyze the data.  In the AUG regression model we definelarge pi = p(Y_CG  Y_PG)Note that the value of large pi greater than .5 means that women in the PG had a greater reduction of alcohol drinks than those in the CG. For statistical results, all p-values < .05 were considered statistically significant and 95% CIs were presented.We first fit an AUC regression model including both main effects of the covariates.  Note that the main effects of the covariates in fact represented their interactions with the GROUP variable, which is different than the linear or generalized linear model frame.  The reason is that the GROUP variable is involved in defining the AUC.  Tables below present the parameter estimates, SEs, p-values, and 95% CIs for model with one and two covariates.  Because parameter beta_2 was not significantly different from 0, we dropped OVITAMIN and fit another model including only the SMOKE main effect.Table below shows a significant interaction between SMOKE and GROUP because the SMOKE was statistically significant (95% CI: (0.06, 1.47)). Therefore, the final model waslogit(hatpi_Smoke) = hatbeta_0 + hatbeta_1*I(Smoke =Yes)Because the interaction between SMOKE and GROUP was significant, we need to use AUC as a measure of the GROUP effect on CHANGE_DRINK for smokers and non-smokers separately using following formula for example for smokers;hatpi_Smoke = frace^hatbeta_0 + hatbeta_1*Smoke =Yes1 + e^hatbeta_0 + hatbeta_1*Smoke =YesSpecifically, the AUCs were 0.537 (insignificant) and 0.713 (significant) for non-smokers and smokers, respectively.  This implies that the effect of positive and control brochures were similar for nonsmokers; however, for smokers, the probability that the positive brochure had a better effect than the control brochure in terms of alcohol reduction is 71.30%, indicating the positive brochure is a better option than the control brochure."
},

{
    "location": "example-python.html#Result-of-sAUC-Regression-with-one-discrete-covariate-1",
    "page": "saucpy in Python",
    "title": "Result of sAUC Regression with one discrete covariate",
    "category": "section",
    "text": "from pandas import read_csv\nfrom saucpy import sAUC\n\n# Data analysis\nfasd = read_csv(\"../saucpy/data/fasd.csv\")\nfasd['group'] = fasd['group'].astype('category')\nfasd['x1']    = fasd['x1'].astype('category')\nfasd['x2']    = fasd['x2'].astype('category')\n\nsAUC.semiparametricAUC(response = \"y\", treatment_group = [\"group\"], input_covariates = [\"x1\"], data = fasd)"
},

{
    "location": "example-python.html#Model-Summary:-one-predictor-1",
    "page": "saucpy in Python",
    "title": "Model Summary: one predictor",
    "category": "section",
    "text": "Predictors Coefficients Std. Error 2.5% 97.5% p\nIntercept -0.909936 0.315218 -1.527751 -0.292121 0.003893\nx1[T.2] 0.766786 0.356420 0.068216 1.465356 0.031448"
},

{
    "location": "example-python.html#Result-of-sAUC-Regression-with-two-discrete-covariates-1",
    "page": "saucpy in Python",
    "title": "Result of sAUC Regression with two discrete covariates",
    "category": "section",
    "text": "sAUC.semiparametricAUC(response = \"y\", treatment_group = [\"group\"], input_covariates = [\"x1\",\"x2\"], data = fasd)"
},

{
    "location": "example-python.html#Model-Summary-:-two-predictors-1",
    "page": "saucpy in Python",
    "title": "Model Summary : two predictors",
    "category": "section",
    "text": "Predictors Coefficients Std. Error 2.5% 97.5% p\nIntercept -1.125352 0.412748 -1.934324 -0.316380 0.006401\nx1[T.2] 0.781264 0.355340 0.084809 1.477718 0.027904\nx2[T.2] 0.252050 0.328229 -0.391267 0.895368 0.442541"
},

{
    "location": "bohora.html#",
    "page": "Developer",
    "title": "Developer",
    "category": "page",
    "text": "About meI am Som and am a Biostatistician at The Department of Pediatrics, The University of Oklahoma Health Sciences Center. I received my MApStat and MS in Biostatistics from LSU and OUHSC, respectively. In addition to BBMC, I work as a statistician and data programmer in a number of pediatric research projects. I was trained in biostatistics and epidemiology, and has research experience in Fetal Alcohol Spectrum Disorders (FASD), HIV/AIDS clinical trials and child maltreatment prevention. I am interested in the applications of statistical computing and simulation, data analytics, dynamic reporting, and real-time data decision making. I use mainly R, Python, and Julia programming languages. "
},

{
    "location": "bug-report.html#",
    "page": "Report Bugs",
    "title": "Report Bugs",
    "category": "page",
    "text": "Report any bugs and queries.You can report any bugs and queries about the package by creating an issue in GitHub package site or by emailing me at som-bohora@ouhsc.edu."
},

]}
