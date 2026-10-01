using ExaModels, NLPModels
# STAGE-3 ACCEPTANCE: the same expression over two DIFFERENT variable blocks
# (different offsets baked in the trees) must MERGE into one family block at
# ExaModel time, with values/derivatives identical to the unmerged model.
c = ExaCore()
c, x = add_var(c, 10; start = 0.3)
c, y = add_var(c, 10; start = 0.8)
c, _ = add_con(c, sin(x[i]) * x[i+1] - 0.1 for i in 1:9; lcon = -2.0, ucon = 2.0)
c, _ = add_con(c, sin(y[i]) * y[i+1] - 0.1 for i in 1:9; lcon = -2.0, ucon = 2.0)
# a third block with a DIFFERENT scalar coefficient: also same type, merges via hoisting
c, _ = add_con(c, sin(x[i]) * x[i+1] - 0.7 for i in 1:9; lcon = -2.0, ucon = 2.0)
c, _ = add_obj(c, abs2(x[i] - y[i]) for i in 1:10)
m = ExaModel(c)
println("blocks: ", length(m.cons), " (expect 1 merged family)")
@assert length(m.cons) == 1
x0 = copy(m.meta.x0)
cv = zeros(m.meta.ncon); NLPModels.cons!(m, x0, cv)
r1 = [sin(0.3)*0.3 - 0.1 for _ in 1:9]
r2 = [sin(0.8)*0.8 - 0.1 for _ in 1:9]
r3 = [sin(0.3)*0.3 - 0.7 for _ in 1:9]
@assert isapprox(cv[1:9], r1; atol=1e-15) "x-block rows"
@assert isapprox(cv[10:18], r2; atol=1e-15) "y-block rows"
@assert isapprox(cv[19:27], r3; atol=1e-15) "coefficient-hoisted rows"
jr = zeros(Int, m.meta.nnzj); jc = zeros(Int, m.meta.nnzj); jv = zeros(m.meta.nnzj)
NLPModels.jac_structure!(m, jr, jc); NLPModels.jac_coord!(m, x0, jv)
@assert all(c -> 11 <= c <= 20, jc[(jr .>= 10) .& (jr .<= 18)]) "y rows hit y columns"
@assert all(c -> 1 <= c <= 10, jc[jr .<= 9]) "x rows hit x columns"
# derivative values: d/dx_i = cos(x_i)*x_{i+1}, d/dx_{i+1} = sin(x_i)
expected = Set([round(cos(0.3)*0.3, digits=12), round(sin(0.3), digits=12),
                round(cos(0.8)*0.8, digits=12), round(sin(0.8), digits=12)])
@assert Set(round.(jv, digits=12)) == expected "jacobian values"
hv = zeros(m.meta.nnzh); hr = zeros(Int, m.meta.nnzh); hc = zeros(Int, m.meta.nnzh)
NLPModels.hess_structure!(m, hr, hc)
NLPModels.hess_coord!(m, x0, ones(m.meta.ncon), hv)
@assert any(!=(0.0), hv)
println("CROSSVAR MERGE OK: 3 same-type blocks (2 variable sets + 1 coefficient variant) merged into 1 family; cons/jac values and columns correct")
