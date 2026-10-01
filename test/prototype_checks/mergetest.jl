using ExaModels, NLPModels
# A: four same-structure blocks added separately (prototype should merge them)
# B: ground truth, one block over the concatenated iterator
rs = [1:100, 201:300, 101:200, 351:450]
function build(ranges_as_blocks::Bool)
    c = ExaCore()
    c, x = add_var(c, 500; start = 0.5)
    c, _ = add_obj(c, abs2(x[i] - 1.0) for i in 1:499)
    if ranges_as_blocks
        for r in rs
            c, _ = add_con(c, sin(x[i]) * cos(x[i+1]) - 0.1 for i in r; lcon = -5.0, ucon = 5.0)
        end
    else
        c, _ = add_con(c, sin(x[i]) * cos(x[i+1]) - 0.1 for i in vcat(collect.(rs)...); lcon = -5.0, ucon = 5.0)
    end
    ExaModel(c)
end
mA = build(true); mB = build(false)
println("nblocks A=", length(mA.cons), " B=", length(mB.cons))
@assert mA.meta.ncon == mB.meta.ncon && mA.meta.nnzj == mB.meta.nnzj && mA.meta.nnzh == mB.meta.nnzh
x0 = copy(mA.meta.x0)
for (name, fA, fB) in (
    ("obj", () -> NLPModels.obj(mA, x0), () -> NLPModels.obj(mB, x0)),)
    @assert fA() == fB()
end
gA = zeros(500); gB = zeros(500)
NLPModels.grad!(mA, x0, gA); NLPModels.grad!(mB, x0, gB); @assert gA == gB
cA = zeros(mA.meta.ncon); cB = zeros(mB.meta.ncon)
NLPModels.cons!(mA, x0, cA); NLPModels.cons!(mB, x0, cB); @assert cA == cB
jrA = zeros(Int, mA.meta.nnzj); jcA = similar(jrA); jvA = zeros(mA.meta.nnzj)
jrB = zeros(Int, mB.meta.nnzj); jcB = similar(jrB); jvB = zeros(mB.meta.nnzj)
NLPModels.jac_structure!(mA, jrA, jcA); NLPModels.jac_structure!(mB, jrB, jcB)
NLPModels.jac_coord!(mA, x0, jvA); NLPModels.jac_coord!(mB, x0, jvB)
@assert jrA == jrB && jcA == jcB && jvA == jvB
hrA = zeros(Int, mA.meta.nnzh); hcA = similar(hrA); hvA = zeros(mA.meta.nnzh)
hrB = zeros(Int, mB.meta.nnzh); hcB = similar(hrB); hvB = zeros(mB.meta.nnzh)
yA = collect(1.0:mA.meta.ncon) ./ mA.meta.ncon
NLPModels.hess_structure!(mA, hrA, hcA); NLPModels.hess_structure!(mB, hrB, hcB)
NLPModels.hess_coord!(mA, x0, yA, hvA); NLPModels.hess_coord!(mB, x0, yA, hvB)
@assert hrA == hrB && hcA == hcB && hvA == hvB
# sanity: the assertions CAN fail — perturb one value and confirm mismatch detection
jvB[1] += 1.0; @assert jvA != jvB
println("MERGE CORRECTNESS OK: merged model bitwise-matches single-block ground truth (obj, grad, cons, jac struct+vals, hess struct+vals)")
