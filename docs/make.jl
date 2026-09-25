using Documenter
using PolyStep

makedocs(;
    sitename = "PolyStep.jl",
    modules = [PolyStep],
    authors = "An T. Le and contributors",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/anindex/PolyStep.jl.git", push_preview = true)
