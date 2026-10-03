using Documenter, DocumenterCitations, ExaModels, Literate

if !(@isdefined _LATEX)
    const _LATEX = true
end

if !(@isdefined _PAGES)
    const _PAGES = [
        "Introduction" => "index.md",
        "Mathematical Abstraction" => "simd.md",
        "Tutorial" => [
            "guide.md",
            "patterns.md",
            "recipe.md",
            "constraint_augmentation.md",
            "performance.md",
            "gpu.md",
            "parameters.md",

            "develop.md",
            "quad.md",
            "distillation.md",
            "opf.md",
            "upgrade.md",
        ],
        "Oracle Constraints" => "oracle.md",
        "JuMP Interface (experimental)" => "jump.md",
        "API Manual" => "core.md",
        "References" => "ref.md",
    ]
end

if !(@isdefined _JL_FILENAMES)
    const _JL_FILENAMES = [
        "guide.jl",
        "recipe.jl",
        "jump.jl",
        "quad.jl",
        "distillation.jl",
        "opf.jl",
        "gpu.jl",
        "performance.jl",
        "parameters.jl",
        "two_stage.jl",
        "oracle.jl",
    ]
end

for jl_filename in _JL_FILENAMES

    Literate.markdown(
        joinpath(@__DIR__, "src", jl_filename),
        joinpath(@__DIR__, "src");
        documenter = true,
        execute = true,
    )

end

bib = CitationBibliography(joinpath(@__DIR__, "src", "refs.bib"))

# if _LATEX
#     makedocs(
#         bib, 
#         sitename = "ExaModels",
#         authors = "Sungho Shin",
#         format = Documenter.LaTeX(),
#         pages = _PAGES,
#     )
# end


makedocs(
    plugins = [bib],
    sitename = "ExaModels.jl",
    modules = [ExaModels],
    authors = "Sungho Shin",
    format = Documenter.HTML(
        assets = ["assets/favicon.ico", "assets/citations.css"],
        prettyurls = true,
        sidebar_sitename = true,
        collapselevel = 1,
    ),
    pages = _PAGES,
    clean = false,
)


deploydocs(repo = "github.com/madsuite-org/ExaModels.jl.git"; push_preview = true)
