using ExaModels, NLPModels
# Correctness: spliced vs lift=true at depth 2, k=2 patterns.
# Lifted model evaluated at [x; e1(x); e2(e1)] must reproduce the spliced
# model's constraint rows, with defining rows at zero.
function build(mode)
    c = ExaCore()
    c, x = add_var(c, 10; start = 1.0)
    c, e1 = add_expr(c, (sin(x[j]) + cos(x[j]) for j in 1:10); lift = mode)
    c, e2 = add_expr(c, (sin(e1[j]) + cos(e1[j]) for j in 1:10); lift = mode)
    c, _ = add_con(c, (exp(e2[j]) + abs2(e2[j+1]) for j in 1:9); lcon = 0.0, ucon = 0.0)
    c, _ = add_con(c, (tanh(e2[j]) * e2[j+1] for j in 1:9); lcon = 0.0, ucon = 0.0)
    c, _ = add_obj(c, x[1])
    ExaModel(c)
end
mS = build(false); mL = build(true)
x0 = fill(0.7, 10)
e1v = sin.(x0) .+ cos.(x0); e2v = sin.(e1v) .+ cos.(e1v)
cS = NLPModels.cons(mS, x0)
cL = NLPModels.cons(mL, vcat(x0, e1v, e2v))
def = cL[1:20]; pat = cL[21:end]
@assert maximum(abs, def) < 1e-14
@assert maximum(abs, pat .- cS) < 1e-14
@assert !(maximum(abs, NLPModels.cons(mL, vcat(x0, e1v .+ 0.1, e2v))[1:20]) < 1e-14)  # falsifiable
println("LIFT CORRECTNESS OK: nvar $(mS.meta.nvar)->$(mL.meta.nvar), ncon $(mS.meta.ncon)->$(mL.meta.ncon); pattern rows match spliced to 1e-14, defining rows zero")
