# Build and evaluate with plain top-level calls, as from a script. `builder` and `backend`
# are set by probe.jl. The starting point comes from `meta` and buffers from `similar`:
# probe.jl says why.
m = builder(20; backend = backend)
x = m.meta.x0
y = m.meta.y0
ncon = NLPModels.get_ncon(m)
nnzj = NLPModels.get_nnzj(m)
nnzh = NLPModels.get_nnzh(m)
NLPModels.jac_structure!(m, similar(x, Int, nnzj), similar(x, Int, nnzj))
NLPModels.hess_structure!(m, similar(x, Int, nnzh), similar(x, Int, nnzh))
NLPModels.obj(m, x)
NLPModels.grad!(m, x, similar(x))
NLPModels.cons!(m, x, similar(x, ncon))
NLPModels.jac_coord!(m, x, similar(x, nnzj))
NLPModels.hess_coord!(m, x, similar(x, nnzh))
NLPModels.hess_coord!(m, x, similar(x, nnzh); obj_weight = 1.0)
NLPModels.hess_coord!(m, x, y, similar(x, nnzh))
NLPModels.hess_coord!(m, x, y, similar(x, nnzh); obj_weight = 1.0)
NLPModels.jac_coord!(m, x, App.padded_view(x, nnzj))
NLPModels.hess_coord!(m, x, y, App.padded_view(x, nnzh); obj_weight = 1.0)
