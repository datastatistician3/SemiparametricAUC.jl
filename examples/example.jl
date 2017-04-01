using DataFrames
using SemiparametricAUC
# Data analysis examples
fasd = DataFrames.readtable(joinpath(Pkg.dir("SemiparametricAUC"), "data/fasd.csv"))
# fasd = readtable("ds.csv")

# Define factor/categorical variable
function convert_to_factor(x)
    return(DataFrames.pool(x))
end

fasd[:group]    = convert_to_factor(fasd[:group])
fasd[:x1]    = convert_to_factor(fasd[:x1])
fasd[:x2]    = convert_to_factor(fasd[:x2])
# fasd[:x3]    = convert_to_factor(fasd[:x3])

SemiparametricAUC.sAUC(x = y ~ x1, treatment_group = :group, data = fasd)
SemiparametricAUC.sAUC(x = y ~ x1 + x2, treatment_group = :group, data = fasd)
