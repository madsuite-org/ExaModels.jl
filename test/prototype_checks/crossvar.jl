using ExaModels, NLPModels
# Same expression over two DIFFERENT variable blocks: do they merge?
c = ExaCore()
c, x = add_var(c, 10; start = 0.3)
c, y = add_var(c, 10; start = 0.8)   # different block, different offsets
c, _ = add_con(c, sin(x[i]) * x[i+1] - 0.1 for i in 1:9; lcon = -2.0, ucon = 2.0)
c, _ = add_con(c, sin(y[i]) * y[i+1] - 0.1 for i in 1:9; lcon = -2.0, ucon = 2.0)
c, _ = add_obj(c, abs2(x[i] - y[i]) for i in 1:10)
m = ExaModel(c)
println("blocks after adding same expression over x and over y: ", length(m.cons)); @assert length(m.cons) == 2
# ground truth: unmerged values
x0 = copy(m.meta.x0)
cv = zeros(m.meta.ncon); NLPModels.cons!(m, x0, cv)
ref = [sin(0.3)*0.3 - 0.1 for _ in 1:9]; ref2 = [sin(0.8)*0.8 - 0.1 for _ in 1:9]
@assert isapprox(cv[1:9], ref; atol=1e-14) && isapprox(cv[10:18], ref2; atol=1e-14)
jv = zeros(m.meta.nnzj); jr = zeros(Int, m.meta.nnzj); jc = zeros(Int, m.meta.nnzj)
NLPModels.jac_structure!(m, jr, jc); NLPModels.jac_coord!(m, x0, jv)
# rows 10..18 must reference the y block's columns (11..20)
@assert all(c -> 11 <= c <= 20, jc[jr .>= 10]) && all(c -> 1 <= c <= 10, jc[jr .<= 9])
println("CROSSVAR OK: values and jacobian columns correct for both variable blocks")
println("(expected: 2 blocks, no merge — same type but different variable offsets)")
