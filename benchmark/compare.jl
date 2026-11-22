using Printf

function read_results(dir, pattern)
    results = Dict{Tuple{String, String, String}, NamedTuple}()
    for f in readdir(dir; join = true)
        endswith(f, ".csv") || continue
        occursin(pattern, basename(f)) || continue
        for line in readlines(f)
            startswith(line, "backend") && continue  # skip header
            parts = split(line, ',')
            length(parts) == 10 || continue
            backend, instance, param = parts[1], parts[2], parts[3]
            nvar = parse(Int, parts[4])
            ncon = parse(Int, parts[5])
            tobj = parse(Float64, parts[6])
            tcon = parse(Float64, parts[7])
            tgrad = parse(Float64, parts[8])
            tjac = parse(Float64, parts[9])
            thess = parse(Float64, parts[10])
            results[(backend, instance, param)] =
                (nvar = nvar, ncon = ncon, tobj = tobj, tcon = tcon, tgrad = tgrad, tjac = tjac, thess = thess)
        end
    end
    return results
end

main_res = read_results(@__DIR__, "main")
current_res = read_results(@__DIR__, "current")

isempty(main_res) && error("No main benchmark results found — run benchmarks first")
isempty(current_res) && error("No current benchmark results found — run benchmarks first")

# Sort by backend (GPU first), then instance, then param (numeric)
const BACKEND_ORDER = Dict("CUDA" => 0, "AMDGPU" => 1, "oneAPI" => 2, "Metal" => 3, "nothing" => 4)

function sort_key(k)
    backend, instance, param = k
    border = get(BACKEND_ORDER, backend, 99)
    pnum = something(tryparse(Int, param), 0)
    return (border, instance, pnum, param)
end

all_keys = sort(collect(union(keys(main_res), keys(current_res))); by = sort_key)

function ratio_str(c, m)
    (isnan(c) || isnan(m) || m == 0) && return "     N/A"
    return @sprintf("%8.3f", c / m)
end

const NAME_W = 30
const SEP = "="^(NAME_W + 5 + 5 * 9)

println()
println("Relative timing: current / main  (values < 1.0 are improvements)")
println()
println(SEP)
@printf("  %-*s  | %8s %8s %8s %8s %8s\n", NAME_W, "backend-instance-param", "obj", "cons", "grad", "jac", "hess")
println(SEP)

prev_backend = Ref("")
for key in all_keys
    backend, instance, param = key
    if backend != prev_backend[] && prev_backend[] != ""
        println("-"^length(SEP))
    end
    prev_backend[] = backend

    name = "$backend-$instance-$param"
    if !haskey(main_res, key)
        @printf("  %-*s  |  (no main result)\n", NAME_W, name)
        continue
    end
    if !haskey(current_res, key)
        @printf("  %-*s  |  (no current result)\n", NAME_W, name)
        continue
    end
    m = main_res[key]
    c = current_res[key]
    @printf(
        "  %-*s  | %s %s %s %s %s\n",
        NAME_W, name,
        ratio_str(c.tobj, m.tobj),
        ratio_str(c.tcon, m.tcon),
        ratio_str(c.tgrad, m.tgrad),
        ratio_str(c.tjac, m.tjac),
        ratio_str(c.thess, m.thess),
    )
end

println(SEP)
println()
