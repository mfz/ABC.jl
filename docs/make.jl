using ABC
using Documenter

DocMeta.setdocmeta!(ABC, :DocTestSetup, :(using ABC); recursive=true)

makedocs(;
    modules=[ABC],
    authors="Florian <zink.florian@gmail.com> and contributors",
    repo="https://github.com/mfz/ABC.jl/blob/{commit}{path}#{line}",
    sitename="ABC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://mfz.github.io/ABC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/mfz/ABC.jl",
    devbranch="main",
)
