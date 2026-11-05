# # Performance Tips

# ## Use a function to create a model

# It is always better to use functions to create ExaModels. This in this way, the functions used for specifing objective/constraint functions are not recreated over all over, and thus, we can take advantage of the previously compiled model creation code. Let's consider the following example.

using ExaModels

t = @elapsed begin
    c = ExaCore(concrete = Val(true))
    N = 10
    @add_var(c, x, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    @add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
    @add_con(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2)
    )
    m = ExaModel(c)
end

println("$t seconds elapsed")

# Even at the second call,
t = @elapsed begin
    c = ExaCore(concrete = Val(true))
    N = 10
    @add_var(c, x, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    @add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
    @add_con(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] - x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2)
    )
    m = ExaModel(c)
end

println("$t seconds elapsed")
# the model creation time can be slightly reduced but the compilation time is still quite significant.

# But instead, if you create a function, we can significantly reduce the model creation time.
function luksan_vlcek_model(N)
    c = ExaCore(concrete = Val(true))
    @add_var(c, x, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    @add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i = 2:N)
    @add_con(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
        x[i]exp(x[i] - x[i+1]) - 3 for i = 1:(N-2)
    )
    m = ExaModel(c)
end

t = @elapsed luksan_vlcek_model(N)
println("$t seconds elapsed")

#-

t = @elapsed luksan_vlcek_model(N)
println("$t seconds elapsed")

# So, the model creation time can be essentially nothing. Thus, if you care about the model creation time, always make sure to write a function for creating the model, and do not directly create a model from the REPL.


# ## Make sure your array's eltype is concrete
# In order for ExaModels to run for loops over the array you provided without any overhead caused by type inference, the eltype of the data array should always be a concrete type. Furthermore, this is **required** if you want to run ExaModels on GPU accelerators.

# Let's take an example.
using ExaModels

N = 1000

function luksan_vlcek_model_concrete(N)
    c = ExaCore(concrete = Val(true))

    arr1 = Array(2:N)
    arr2 = Array(1:(N-2))

    @add_var(c, x, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    @add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i in arr1)
    @add_con(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
        x[i]exp(x[i] - x[i+1]) - 3 for i in arr2
    )
    m = ExaModel(c)
end

function luksan_vlcek_model_non_concrete(N)
    c = ExaCore(concrete = Val(true))

    arr1 = Array{Any}(2:N)
    arr2 = Array{Any}(1:(N-2))

    @add_var(c, x, N; start = (mod(i, 2) == 1 ? -1.2 : 1.0 for i = 1:N))
    @add_obj(c, 100 * (x[i-1]^2 - x[i])^2 + (x[i-1] - 1)^2 for i in arr1)
    @add_con(
        c,
        3x[i+1]^3 + 2 * x[i+2] - 5 + sin(x[i+1] - x[i+2])sin(x[i+1] + x[i+2]) + 4x[i+1] -
        x[i]exp(x[i] - x[i+1]) - 3 for i in arr2
    )
    m = ExaModel(c)
end


# Here, observe that

isconcretetype(eltype(Array(2:N)))
#-

isconcretetype(eltype(Array{Any}(2:N)))

# As you can see, the first array type has concrete eltypes, whereas the second array type has non concrete eltypes. Due to this, the array stored in the model created by `luksan_vlcek_model_non_concrete` will have non-concrete eltypes.

# Now let's compare the performance. We will use the following benchmark function here.

using NLPModels

function benchmark_callbacks(m; N = 100)
    nvar = m.meta.nvar
    ncon = m.meta.ncon
    nnzj = m.meta.nnzj
    nnzh = m.meta.nnzh

    x = copy(m.meta.x0)
    y = similar(m.meta.x0, ncon)
    c = similar(m.meta.x0, ncon)
    g = similar(m.meta.x0, nvar)
    jac = similar(m.meta.x0, nnzj)
    hess = similar(m.meta.x0, nnzh)
    jrows = similar(m.meta.x0, Int, nnzj)
    jcols = similar(m.meta.x0, Int, nnzj)
    hrows = similar(m.meta.x0, Int, nnzh)
    hcols = similar(m.meta.x0, Int, nnzh)

    GC.enable(false)

    NLPModels.obj(m, x) # to compile

    tobj = (1 / N) * @elapsed for t = 1:N
        NLPModels.obj(m, x)
    end

    NLPModels.cons!(m, x, c) # to compile
    tcon = (1 / N) * @elapsed for t = 1:N
        NLPModels.cons!(m, x, c)
    end

    NLPModels.grad!(m, x, g) # to compile
    tgrad = (1 / N) * @elapsed for t = 1:N
        NLPModels.grad!(m, x, g)
    end

    NLPModels.jac_coord!(m, x, jac) # to compile
    tjac = (1 / N) * @elapsed for t = 1:N
        NLPModels.jac_coord!(m, x, jac)
    end

    NLPModels.hess_coord!(m, x, y, hess) # to compile
    thess = (1 / N) * @elapsed for t = 1:N
        NLPModels.hess_coord!(m, x, y, hess)
    end

    NLPModels.jac_structure!(m, jrows, jcols) # to compile
    tjacs = (1 / N) * @elapsed for t = 1:N
        NLPModels.jac_structure!(m, jrows, jcols)
    end

    NLPModels.hess_structure!(m, hrows, hcols) # to compile
    thesss = (1 / N) * @elapsed for t = 1:N
        NLPModels.hess_structure!(m, hrows, hcols)
    end

    GC.enable(true)

    return (
        tobj = tobj,
        tcon = tcon,
        tgrad = tgrad,
        tjac = tjac,
        thess = thess,
        tjacs = tjacs,
        thesss = thesss,
    )
end

# The performance comparison is here:
m1 = luksan_vlcek_model_concrete(N)
m2 = luksan_vlcek_model_non_concrete(N)

benchmark_callbacks(m1)

#-

benchmark_callbacks(m2)

# As can be seen here, having concrete eltype dramatically improves the performance. This is because when all the data arrays' eltypes are concrete, the AD evaluations can be performed without any type inferernce, and this should be as fast as highly optimized C/C++/Fortran code.

# When you're using GPU accelerators, the eltype of the array should always be concrete. In fact, non-concrete etlype will already cause an error when creating the array. For example,

using CUDA

try
    arr1 = CuArray(Array{Any}(2:N))
catch e
    showerror(stdout, e)
end
