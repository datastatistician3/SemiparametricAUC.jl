using Documenter, SemiparametricAUC

makedocs(
  modules  = [SemiparametricAUC],
  clean    = false,
  format   = :html,
  sitename = :"Perform semiparametric AUC model in Julia",
  authors  = "Som B. Bohora",
  # assets   = ["assets/favicon.ico"],
  pages    = [
              "Introduction"    => "intro.md",
              "Installation"    => "install.md",
              "Article"         => "paper.md",
              "Example"         => "example-julia.md",
              "References"        => "index.md",
              "Code"            => "code.md",
              "sAUC in R"       => "example.md",
              "sAUC in R Shiny" => "r-shiny.md",
              "saucpy in Python"=> "example-python.md",
              "Developer"       => "bohora.md",
              "Report Bugs"     => "bug-report.md"
  ]
)

deploydocs(
    deps   = Deps.pip("mkdocs", "python-markdown-math"),
    target = "build",
    repo   = "github.com/sbohora/SemiparametricAUC.jl.git",
    make   = nothing,
    julia  = "release",
    osname = "osx"
)
