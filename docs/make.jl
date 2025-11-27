using Documenter
using SuperLU

makedocs(
    sitename = "SuperLU.jl",
    modules = [SuperLU],
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true",
        canonical = "https://TendonFFF.github.io/SuperLU.jl/stable",
    ),
    pages = [
        "Home" => "index.md",
        "Getting Started" => "getting_started.md",
        "Solver Options" => "options.md",
        "API Reference" => "api.md",
    ],
    warnonly = [:missing_docs],
)

deploydocs(
    repo = "github.com/TendonFFF/SuperLU.jl.git",
    devbranch = "main",
    push_preview = true,
)
