"""
    BundleDepot

The first package of a bundled library's image (`compile_library(...; bundle = true)`): when
the library starts, its runtime looks for artifacts only in the bundle's own depot
(share/julia), not in a Julia installation's (~/.julia, JULIA_DEPOT_PATH). A user's depot may
hold the same artifacts by hash under their standard library names, and loading those would
tie this runtime to libraries another runtime in the process uses. Its `__init__` runs before
any other package's, since the library module imports it first.
`CNLP_DEPOT_TRACE=1` prints the depot path before and after.
"""
module BundleDepot

function __init__()
    ccall(:jl_generating_output, Cint, ()) == 0 || return   # not while the image is built
    trace = get(ENV, "CNLP_DEPOT_TRACE", "0") == "1"
    # Core.stderr, a constant: the call resolves in a trimmed image (Base's stderr does not)
    trace && Core.println(Core.stderr, "depot path ", join(DEPOT_PATH, ":"))
    root = normpath(joinpath(Sys.BINDIR, ".."))
    keep = filter(d -> startswith(normpath(abspath(d)), root), DEPOT_PATH)
    isempty(keep) || (empty!(DEPOT_PATH); append!(DEPOT_PATH, keep))
    trace && Core.println(Core.stderr, "depot path (bundle only) ", join(DEPOT_PATH, ":"))
    return nothing
end

end # module
