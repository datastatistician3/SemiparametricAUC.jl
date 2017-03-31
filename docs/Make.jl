using Documenter, SemiparametricAUC

makedocs()

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/sbohora/SemiparametricAUC.jl.git",
    julia  = "0.5",
    osname = "osx"
)
