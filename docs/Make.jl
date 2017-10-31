using Documenter, SemiparametricAUC

makedocs(
  format = :html,
  sitename = :"SemiparametricAUC.jl",
  pages = [
      "First title page" => "sec2/page1.md",
      "Second Page title" => "tutorial/page1.md"
  ]
)

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    repo   = "github.com/sbohora/SemiparametricAUC.jl.git",
    julia  = "0.5",
    osname = "osx"
)
