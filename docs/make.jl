using Documenter
using SuperLU

makedocs(
    sitename = "SuperLU.jl",
    format = Documenter.HTML(
        prettyurls = get(ENV, "CI", nothing) == "true"
    ),
    modules = [SuperLU],
    pages = [
        "Home" => "index.md",
    ],
)

deploydocs(
    repo = "github.com/TendonFFF/SuperLU.jl.git",
    devbranch = "main",
    push_preview = true,
)
