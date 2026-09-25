using Documenter
using PolyStep

const REPO = "https://github.com/anindex/PolyStep.jl/blob/main/"
readme = read(joinpath(@__DIR__, "..", "README.md"), String)
readme = replace(readme, r"\]\((?!https?://|#)([^)]+)\)" => SubstitutionString("]($(REPO)\\1)"))
readme = replace(readme, r"https://anindex\.github\.io/PolyStep\.jl/stable/(\w+)/" => s"\1.md")
write(joinpath(@__DIR__, "src", "index.md"), readme)

makedocs(;
    sitename = "PolyStep.jl",
    modules = [PolyStep],
    authors = "An T. Le and contributors",
    format = Documenter.HTML(; prettyurls = get(ENV, "CI", "false") == "true"),
    pages = [
        "Home" => "index.md",
        "Guide" => "guide.md",
        "Benchmarks" => "benchmarks.md",
        "API" => "api.md",
    ],
    checkdocs = :exports,
)

deploydocs(; repo = "github.com/anindex/PolyStep.jl.git", push_preview = true)
