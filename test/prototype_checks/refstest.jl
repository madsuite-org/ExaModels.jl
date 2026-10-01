using ExaModels, NLPModels
c = ExaCore()
@add_var(c, x, 10; start = 0.5)
@add_con(c, g1, sin(x[i]) - 0.1 for i in 1:9; lcon = -1.0, ucon = 1.0)
c, p = add_par(c, 3; name = Val(:price))
@add_obj(c, obj1, abs2(x[i] - 1.0) for i in 1:10)
# refs must be Vector{Pair} in default (non-concrete) mode
@assert getfield(c, :refs) isa Vector{Pair{Symbol, Any}}
# core.name lookup through the vector
@assert c.g1 isa ExaModels.Constraint
@assert c.x isa ExaModels.Variable
@assert length(get_cons(c)) == 1 && haskey(get_cons(c), :g1)
@assert get_vars(c, :x) === c.x
m = ExaModel(c)
# model refs materialized to NamedTuple; model.name works
@assert getfield(m, :refs) isa NamedTuple
@assert m.g1 isa ExaModels.Constraint
@assert haskey(get_cons(m), :g1) && haskey(get_vars(m), :x)
# name overwrite semantics
c2 = ExaCore()
c2, a = add_var(c2, 2; name = Val(:v))
c2, b = add_var(c2, 3; name = Val(:v))
@assert c2.v === b
# concrete mode unchanged (NamedTuple end to end)
c3 = ExaCore(concrete = Val(true))
c3, y = add_var(c3, 4; name = Val(:y))
@assert getfield(c3, :refs) isa NamedTuple && c3.y === y
# error path still informative
try get_vars(m, :g1); @assert false catch e; @assert occursin("constraint", sprint(showerror, e)) end
println("REFS OK")
