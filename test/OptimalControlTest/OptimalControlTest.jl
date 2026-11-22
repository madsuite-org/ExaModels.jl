module OptimalControlTest

using ExaModels, OptimalControl
using Test, ForwardDiff, LinearAlgebra

# # --- AD correctness helpers (from original LinAlgTest) ---

# function gradient(f, x)
#     T = eltype(x)
#     y = fill!(similar(x), zero(T))
#     ExaModels.gradient!(y, (p, x, θ) -> f(x), x, nothing, nothing, one(T))
#     return y
# end

# function sgradient(f, x)
#     T = eltype(x)

#     ff = f(ExaModels.VarSource())
#     d = ff(ExaModels.Identity(), ExaModels.AdjointNodeSource(nothing), nothing)
#     y1 = []
#     ExaModels.grpass(d, nothing, y1, nothing, 0, NaN)

#     a1 = unique(y1)
#     comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a1) for i in y1))

#     n = length(a1)
#     buffer = fill!(similar(x, n), zero(T))
#     buffer_I = similar(x, Tuple{Int, Int}, n)

#     ExaModels.sgradient!(buffer_I, ff, nothing, nothing, nothing, comp, 0, NaN)
#     ExaModels.sgradient!(buffer, ff, nothing, x, nothing, comp, 0, one(T))

#     y = zeros(length(x))
#     y[collect(i for (i, j) in buffer_I)] += buffer

#     return y
# end

# function shessian(f, x)
#     T = eltype(x)

#     ff = f(ExaModels.VarSource())
#     t = ff(ExaModels.Identity(), ExaModels.SecondAdjointNodeSource(nothing), nothing)
#     y2 = []
#     ExaModels.hrpass0(t, nothing, y2, nothing, nothing, 0, NaN, NaN)

#     a2 = unique(y2)
#     comp = ExaModels.Compressor(Tuple(findfirst(isequal(i), a2) for i in y2))

#     n = length(a2)
#     buffer = fill!(similar(x, n), zero(T))
#     buffer_I = similar(x, Int, n)
#     buffer_J = similar(x, Int, n)

#     ExaModels.shessian!(
#         buffer_I,
#         buffer_J,
#         ff,
#         nothing,
#         nothing,
#         nothing,
#         comp,
#         0,
#         NaN,
#         NaN,
#     )
#     ExaModels.shessian!(buffer, nothing, ff, nothing, x, nothing, comp, 0, one(T), zero(T))

#     y = zeros(length(x), length(x))
#     for (k, (i, j)) in enumerate(zip(buffer_I, buffer_J))
#         if i == j
#             y[i, j] += buffer[k]
#         else
#             y[i, j] += buffer[k]
#             y[j, i] += buffer[k]
#         end
#     end
#     return y
# end

_vec(x, inds) = [x[i] for i in inds]
_mat(x, rows, cols) = [x[rows[i] + cols[j] - 1] for i in eachindex(rows), j in eachindex(cols)]

const LINALG_FUNCTIONS = [
    ("linalg-dot-node-node", x -> dot(_vec(x, 1:3), _vec(x, 4:6))),
    ("linalg-dot-real-node", x -> dot([1.0, 2.0, 3.0], _vec(x, 1:3))),
    ("linalg-dot-node-real", x -> dot(_vec(x, 1:3), [1.0, 2.0, 3.0])),
    ("linalg-sum", x -> sum(_vec(x, 1:4))),
    ("linalg-norm2", x -> norm(_vec(x, 1:3))),
    ("linalg-norm3", x -> norm(_vec(x, 1:3), 3)),
    (
        "linalg-matvec-node-node",
        x -> begin
            A = [x[1] x[3]; x[2] x[4]]
            v = [x[5], x[6]]
            r = A * v
            r[1] + r[2]
        end,
    ),
    (
        "linalg-matvec-real-node",
        x -> begin
            A = [1.0 2.0; 3.0 4.0]
            v = [x[1], x[2]]
            r = A * v
            r[1] + r[2]
        end,
    ),
    (
        "linalg-tr",
        x -> begin
            A = [x[1] x[3]; x[2] x[4]]
            tr(A)
        end,
    ),
    (
        "linalg-det-2x2",
        x -> begin
            A = [x[1] x[2]; x[3] x[4]]
            det(A)
        end,
    ),
    (
        "linalg-det-3x3",
        x -> begin
            A = [x[1] x[2] x[3]; x[4] x[5] x[6]; x[7] x[8] x[9]]
            det(A)
        end,
    ),
    (
        "linalg-cross",
        x -> begin
            c = cross(_vec(x, 1:3), _vec(x, 4:6))
            c[1] + c[2] + c[3]
        end,
    ),
    (
        "linalg-vec-add",
        x -> begin
            r = _vec(x, 1:3) + _vec(x, 4:6)
            r[1] + r[2] + r[3]
        end,
    ),
    (
        "linalg-vec-sub",
        x -> begin
            r = _vec(x, 1:3) - _vec(x, 4:6)
            r[1] + r[2] + r[3]
        end,
    ),
    (
        "linalg-scalar-vec",
        x -> begin
            r = x[1] * _vec(x, 2:4)
            r[1] + r[2] + r[3]
        end,
    ),
    (
        "linalg-matmul",
        x -> begin
            A = [x[1] x[2]; x[3] x[4]]
            B = [x[5] x[6]; x[7] x[8]]
            C = A * B
            C[1, 1] + C[1, 2] + C[2, 1] + C[2, 2]
        end,
    ),
    (
        "linalg-composite-norm-matvec",
        x -> begin
            A = [1.0 2.0; 3.0 4.0]
            v = [x[1], x[2]]
            norm(A * v)
        end,
    ),
    (
        "linalg-composite-det-dot",
        x -> begin
            A = [x[1] x[2]; x[3] x[4]]
            det(A) * dot(_vec(x, 5:6), _vec(x, 7:8))
        end,
    ),
]

# --- Type dispatch helpers (from original LinAlgTest2) ---

is_null_zero(x::ExaModels.Null) = iszero(x.value)
is_null_zero(x::ExaModels.AbstractNode) = false

function create_nodes()
    x = ExaModels.Null(1.0)
    y = ExaModels.Null(2.0)
    z = ExaModels.Null(3.0)
    w = ExaModels.Null(4.0)
    return x, y, z, w
end

# --- Main test runner ---

function runtests()
    return @testset "Linear Algebra test" begin

        # =====================================================================
        # AD correctness: gradient, sparse gradient, sparse Hessian vs ForwardDiff
        # =====================================================================
        # @testset "AD correctness" begin
        #     for (name, f) in LINALG_FUNCTIONS
        #         x0 = 0.5 .+ rand(10)  # avoid zero for norm derivatives
        #         @testset "$name" begin
        #             g_fd = ForwardDiff.gradient(f, x0)
        #             h_fd = ForwardDiff.hessian(f, x0)
        #             @test gradient(f, x0) ≈ g_fd atol = 1.0e-6
        #             @test sgradient(f, x0) ≈ g_fd atol = 1.0e-6
        #             @test shessian(f, x0) ≈ h_fd atol = 1.0e-6
        #         end
        #     end
        # end

        # =====================================================================
        # Type dispatch and structure: return types, sizes, zero optimizations
        # =====================================================================
        @testset "Type dispatch and structure" begin
            x, y, z, w = create_nodes()

            @testset "Type conversions and promotions" begin
                node = convert(ExaModels.AbstractNode, 5)
                @test node isa ExaModels.Null
                @test node isa ExaModels.AbstractNode
                @test node.value == 5

                zero_int = convert(ExaModels.AbstractNode, 0)
                @test zero_int isa ExaModels.Null
                @test iszero(zero_int.value)
                @test zero_int === zero(ExaModels.AbstractNode)

                zero_float = convert(ExaModels.AbstractNode, 0.0)
                @test zero_float isa ExaModels.Null
                @test iszero(zero_float.value)
                @test zero_float === zero(ExaModels.AbstractNode)

                arr = [x, 2.0, 3.0]
                @test eltype(arr) == ExaModels.AbstractNode
            end

            @testset "Scalar × Vector multiplication" begin
                v_num = [1.0, 2.0, 3.0]

                result1 = x * v_num
                @test length(result1) == 3
                @test result1 isa Vector
                @test result1[1] isa ExaModels.AbstractNode

                vec_nodes = [x, y, z]
                result2 = 2.0 * vec_nodes
                @test length(result2) == 3
                @test result2 isa Vector
                @test result2[1] isa ExaModels.AbstractNode
            end

            @testset "Vector × Scalar multiplication" begin
                v_num = [1.0, 2.0, 3.0]
                vec_nodes = [x, y, z]

                result1 = v_num * x
                @test length(result1) == 3
                @test result1 isa Vector
                @test result1[1] isa ExaModels.AbstractNode

                result2 = vec_nodes .* 2.5
                @test length(result2) == 3
                @test result2 isa Vector
                @test result2[1] isa ExaModels.AbstractNode
            end

            @testset "Scalar × Matrix multiplication" begin
                A_num = [1.0 2.0; 3.0 4.0]
                mat_nodes = [x y; z w]

                result1 = x * A_num
                @test size(result1) == (2, 2)
                @test result1 isa Matrix
                @test result1[1, 1] isa ExaModels.AbstractNode

                result2 = 3.0 * mat_nodes
                @test size(result2) == (2, 2)
                @test result2 isa Matrix
                @test result2[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Matrix × Scalar multiplication" begin
                A_num = [1.0 2.0; 3.0 4.0]
                mat_nodes = [x y; z w]

                result1 = A_num * x
                @test size(result1) == (2, 2)
                @test result1 isa Matrix
                @test result1[1, 1] isa ExaModels.AbstractNode

                result2 = mat_nodes .* 1.5
                @test size(result2) == (2, 2)
                @test result2 isa Matrix
                @test result2[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Dot product" begin
                v_num = [1.0, 2.0, 3.0]
                vec_nodes = [x, y, z]

                result1 = dot(v_num, vec_nodes)
                @test result1 isa ExaModels.AbstractNode

                result2 = dot(vec_nodes, v_num)
                @test result2 isa ExaModels.AbstractNode
            end

            @testset "Dot product (Real × Real fallback)" begin
                v1 = [1.0, 2.0, 3.0]
                v2 = [4.0, 5.0, 6.0]

                result1 = dot(v1, v2)
                @test result1 isa Real
                @test result1 ≈ 32.0

                v1_view = @view v1[1:3]
                v2_view = @view v2[1:3]
                result2 = dot(v1_view, v2_view)
                @test result2 isa Real
                @test result2 ≈ 32.0

                v1_reshaped = reshape([1.0, 2.0, 3.0], 3)
                v2_reshaped = reshape([4.0, 5.0, 6.0], 3)
                result3 = dot(v1_reshaped, v2_reshaped)
                @test result3 isa Real
                @test result3 ≈ 32.0

                complex_vec1 = ComplexF64[1.0 + 2.0im, 3.0 + 4.0im]
                complex_vec2 = ComplexF64[5.0 + 6.0im, 7.0 + 8.0im]
                real_reinterp1 = reinterpret(Float64, complex_vec1)
                real_reinterp2 = reinterpret(Float64, complex_vec2)
                result4 = dot(real_reinterp1, real_reinterp2)
                @test result4 isa Real
                @test result4 ≈ 1 * 5 + 2 * 6 + 3 * 7 + 4 * 8

                result5 = dot(v1_view, v2_reshaped)
                @test result5 isa Real
                @test result5 ≈ 32.0
            end

            @testset "Matrix × Vector product" begin
                A_num = [1.0 2.0 3.0; 4.0 5.0 6.0]
                vec_nodes = [x, y, z]

                result = A_num * vec_nodes
                @test length(result) == 2
                @test result isa Vector
                @test result[1] isa ExaModels.AbstractNode
            end

            @testset "Matrix × Matrix product" begin
                A_num = [1.0 2.0; 3.0 4.0]
                B_nodes = [x y; z w]

                result1 = A_num * B_nodes
                @test size(result1) == (2, 2)
                @test result1 isa Matrix
                @test result1[1, 1] isa ExaModels.AbstractNode

                result2 = B_nodes * A_num
                @test size(result2) == (2, 2)
                @test result2 isa Matrix
                @test result2[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Adjoint Vector × Vector product" begin
                vec_nodes = [x, y, z]
                v_num = [1.0, 2.0, 3.0]

                result1 = vec_nodes' * v_num
                @test result1 isa ExaModels.AbstractNode

                result2 = v_num' * vec_nodes
                @test result2 isa ExaModels.AbstractNode

                vec_nodes2 = [y, z, x]
                result3 = vec_nodes' * vec_nodes2
                @test result3 isa ExaModels.AbstractNode
            end

            @testset "Adjoint Vector × Matrix product" begin
                vec_nodes = [x, y, z]
                A_num = [1.0 2.0; 3.0 4.0; 5.0 6.0]

                result = vec_nodes' * A_num
                @test size(result) == (1, 2)
                @test result isa LinearAlgebra.Adjoint
            end

            @testset "Matrix adjoint" begin
                mat_nodes = [x y z; y z x]

                result = adjoint(mat_nodes)
                @test size(result) == (3, 2)
                @test result isa Matrix
            end

            @testset "Determinant" begin
                A1 = reshape([x], 1, 1)
                result1 = det(A1)
                @test result1 isa ExaModels.AbstractNode

                A2 = [x y; z w]
                result2 = det(A2)
                @test result2 isa ExaModels.AbstractNode

                A3 = [x y z; y z x; z x y]
                result3 = det(A3)
                @test result3 isa ExaModels.AbstractNode
            end

            @testset "Broadcasting operations" begin
                vec_nodes = [x, y, z]
                mat_nodes = [x y; z w]

                result1 = cos.(vec_nodes)
                @test length(result1) == 3
                @test result1 isa Vector
                @test result1[1] isa ExaModels.AbstractNode

                result2 = sin.(vec_nodes)
                @test length(result2) == 3
                @test result2[1] isa ExaModels.AbstractNode

                result3 = exp.(vec_nodes)
                @test length(result3) == 3
                @test result3[1] isa ExaModels.AbstractNode

                result4 = cos.(mat_nodes)
                @test size(result4) == (2, 2)
                @test result4 isa Matrix
                @test result4[1, 1] isa ExaModels.AbstractNode

                result5 = vec_nodes .+ 1.0
                @test length(result5) == 3
                @test result5[1] isa ExaModels.AbstractNode

                result6 = vec_nodes .* 2.0
                @test length(result6) == 3
                @test result6[1] isa ExaModels.AbstractNode

                v_num = [1.0, 2.0, 3.0]
                result7 = vec_nodes .+ v_num
                @test length(result7) == 3
                @test result7[1] isa ExaModels.AbstractNode

                result8 = vec_nodes .* v_num
                @test length(result8) == 3
                @test result8[1] isa ExaModels.AbstractNode
            end

            @testset "Trace" begin
                A2 = [x y; z w]
                result1 = tr(A2)
                @test result1 isa ExaModels.AbstractNode

                A3 = [x y z; y z w; z w x]
                result2 = tr(A3)
                @test result2 isa ExaModels.AbstractNode

                A_rect = [x y z; y z w]
                @test_throws AssertionError tr(A_rect)
            end

            @testset "Norms" begin
                vec_nodes = [x, y, z]
                mat_nodes = [x y; z w]

                result1 = norm(vec_nodes)
                @test result1 isa ExaModels.AbstractNode

                result2 = norm(vec_nodes, 1)
                @test result2 isa ExaModels.AbstractNode

                result3 = norm(vec_nodes, 2)
                @test result3 isa ExaModels.AbstractNode

                result4 = norm(vec_nodes, 3)
                @test result4 isa ExaModels.AbstractNode

                result5 = norm(mat_nodes)
                @test result5 isa ExaModels.AbstractNode

                @test_throws ErrorException norm(vec_nodes, Inf)
            end

            @testset "Array addition" begin
                vec_nodes1 = [x, y, z]
                vec_nodes2 = [y, z, w]
                v_num = [1.0, 2.0, 3.0]

                result1 = vec_nodes1 + v_num
                @test length(result1) == 3
                @test result1 isa Vector
                @test result1[1] isa ExaModels.AbstractNode

                result2 = v_num + vec_nodes1
                @test length(result2) == 3
                @test result2[1] isa ExaModels.AbstractNode

                result3 = vec_nodes1 + vec_nodes2
                @test length(result3) == 3
                @test result3[1] isa ExaModels.AbstractNode

                mat_nodes1 = [x y; z w]
                mat_nodes2 = [y z; w x]
                A_num = [1.0 2.0; 3.0 4.0]

                result4 = mat_nodes1 + A_num
                @test size(result4) == (2, 2)
                @test result4 isa Matrix
                @test result4[1, 1] isa ExaModels.AbstractNode

                result5 = A_num + mat_nodes1
                @test size(result5) == (2, 2)
                @test result5[1, 1] isa ExaModels.AbstractNode

                result6 = mat_nodes1 + mat_nodes2
                @test size(result6) == (2, 2)
                @test result6[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Array subtraction" begin
                vec_nodes1 = [x, y, z]
                vec_nodes2 = [y, z, w]
                v_num = [1.0, 2.0, 3.0]

                result1 = vec_nodes1 - v_num
                @test length(result1) == 3
                @test result1 isa Vector
                @test result1[1] isa ExaModels.AbstractNode

                result2 = v_num - vec_nodes1
                @test length(result2) == 3
                @test result2[1] isa ExaModels.AbstractNode

                result3 = vec_nodes1 - vec_nodes2
                @test length(result3) == 3
                @test result3[1] isa ExaModels.AbstractNode

                mat_nodes1 = [x y; z w]
                mat_nodes2 = [y z; w x]
                A_num = [1.0 2.0; 3.0 4.0]

                result4 = mat_nodes1 - A_num
                @test size(result4) == (2, 2)
                @test result4 isa Matrix
                @test result4[1, 1] isa ExaModels.AbstractNode

                result5 = A_num - mat_nodes1
                @test size(result5) == (2, 2)
                @test result5[1, 1] isa ExaModels.AbstractNode

                result6 = mat_nodes1 - mat_nodes2
                @test size(result6) == (2, 2)
                @test result6[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Diagonal operations" begin
                mat_nodes = [x y z; w x y; z w x]
                result1 = diag(mat_nodes)
                @test length(result1) == 3
                @test result1 isa Vector
                @test result1[1] isa ExaModels.AbstractNode

                mat_rect = [x y z; w x y]
                result2 = diag(mat_rect)
                @test length(result2) == 2
                @test result2[1] isa ExaModels.AbstractNode

                vec_nodes = [x, y, z]
                result3 = diagm(vec_nodes)
                @test size(result3) == (3, 3)
                @test result3 isa Matrix
                @test result3[1, 1] isa ExaModels.AbstractNode
                @test is_null_zero(result3[1, 2])
                @test is_null_zero(result3[2, 1])

                result4 = diagm(1 => vec_nodes)
                @test size(result4) == (4, 4)
                @test result4[1, 2] isa ExaModels.AbstractNode
                @test is_null_zero(result4[1, 1])

                result5 = diagm(-1 => vec_nodes)
                @test size(result5) == (4, 4)
                @test result5[2, 1] isa ExaModels.AbstractNode
                @test is_null_zero(result5[1, 1])
            end

            @testset "Transpose operations" begin
                result1 = transpose(x)
                @test result1 isa ExaModels.AbstractNode

                mat_nodes = [x y z; w x y]
                result2 = transpose(mat_nodes)
                @test size(result2) == (3, 2)
                @test result2 isa Matrix
                @test result2[1, 1] isa ExaModels.AbstractNode
            end

            @testset "Dimension mismatch errors" begin
                v1 = [1.0, 2.0]
                vec_nodes = [x, y, z]
                @test_throws AssertionError dot(v1, vec_nodes)

                A = [1.0 2.0; 3.0 4.0]
                v = [x, y, z]
                @test_throws AssertionError A * v

                A1 = [1.0 2.0; 3.0 4.0]
                A2_nodes = [x y; z w; y x]
                @test_throws AssertionError A1 * A2_nodes

                A_rect = [x y z; y z x]
                @test_throws AssertionError det(A_rect)

                v1 = [x, y]
                v2 = [x, y, z]
                @test_throws AssertionError v1 + v2
                @test_throws AssertionError v1 - v2

                A1 = [x y; z w]
                A2 = [x y z; w x y]
                @test_throws AssertionError A1 + A2
                @test_throws AssertionError A1 - A2
            end

            @testset "ExaCore variable arrays" begin
                c = ExaModels.ExaCore()
                xvar = ExaModels.variable(c, 2, 0:10, lvar = 0, uvar = 1)

                v = [xvar[i, 1] for i in 1:2]
                @test length(v) == 2
                @test v isa Vector
                @test v[1] isa ExaModels.AbstractNode

                A = [xvar[i, j] for (i, j) in Base.product(1:2, 0:3)]
                @test size(A) == (2, 4)
                @test A isa Matrix
                @test A[1, 1] isa ExaModels.AbstractNode

                v_num = [1.0, 2.0]
                result1 = v + v_num
                @test length(result1) == 2
                @test result1[1] isa ExaModels.AbstractNode

                result2 = v - v_num
                @test length(result2) == 2
                @test result2[1] isa ExaModels.AbstractNode

                result3 = 2.0 * v
                @test length(result3) == 2
                @test result3[1] isa ExaModels.AbstractNode

                result4 = dot(v, v_num)
                @test result4 isa ExaModels.AbstractNode

                result5 = norm(v)
                @test result5 isa ExaModels.AbstractNode

                v2 = [xvar[i, 2] for i in 1:2]
                result6 = A * [1.0, 2.0, 3.0, 4.0]
                @test length(result6) == 2
                @test result6[1] isa ExaModels.AbstractNode

                A_square = [xvar[i, j] for (i, j) in Base.product(1:2, 1:2)]
                @test size(A_square) == (2, 2)

                result7 = det(A_square)
                @test result7 isa ExaModels.AbstractNode

                result8 = tr(A_square)
                @test result8 isa ExaModels.AbstractNode

                result9 = diag(A_square)
                @test length(result9) == 2
                @test result9[1] isa ExaModels.AbstractNode

                A_num = [1.0 2.0 3.0 4.0; 5.0 6.0 7.0 8.0]
                result10 = A + A_num
                @test size(result10) == (2, 4)
                @test result10[1, 1] isa ExaModels.AbstractNode

                result11 = A - A_num
                @test size(result11) == (2, 4)
                @test result11[1, 1] isa ExaModels.AbstractNode

                result12 = cos.(v)
                @test length(result12) == 2
                @test result12[1] isa ExaModels.AbstractNode

                result13 = sin.(A_square)
                @test size(result13) == (2, 2)
                @test result13[1, 1] isa ExaModels.AbstractNode

                result14 = v'
                @test size(result14) == (1, 2)
                @test result14 isa LinearAlgebra.Adjoint

                result15 = transpose(A)
                @test size(result15) == (4, 2)
                @test result15 isa Matrix

                result16 = diagm(v)
                @test size(result16) == (2, 2)
                @test result16[1, 1] isa ExaModels.AbstractNode
                @test is_null_zero(result16[1, 2])
            end

            @testset "Method ambiguity fixes" begin
                x, y, z, w = create_nodes()
                vec_nodes1 = [x, y, z]
                vec_nodes2 = [y, z, w]
                mat_nodes1 = [x y; z w]
                mat_nodes2 = [y z; w x]

                @testset "Scalar × Vector (both AbstractNode)" begin
                    result = x * vec_nodes1
                    @test length(result) == 3
                    @test result isa Vector
                    @test result[1] isa ExaModels.AbstractNode
                end

                @testset "Vector × Scalar (both AbstractNode)" begin
                    result = vec_nodes1 * x
                    @test length(result) == 3
                    @test result isa Vector
                    @test result[1] isa ExaModels.AbstractNode
                end

                @testset "Scalar × Matrix (both AbstractNode)" begin
                    result = x * mat_nodes1
                    @test size(result) == (2, 2)
                    @test result isa Matrix
                    @test result[1, 1] isa ExaModels.AbstractNode
                end

                @testset "Matrix × Scalar (both AbstractNode)" begin
                    result = mat_nodes1 * x
                    @test size(result) == (2, 2)
                    @test result isa Matrix
                    @test result[1, 1] isa ExaModels.AbstractNode
                end

                @testset "dot product (both AbstractNode)" begin
                    result = dot(vec_nodes1, vec_nodes2)
                    @test result isa ExaModels.AbstractNode
                end

                @testset "Matrix × Vector (both AbstractNode)" begin
                    A = [x y z; w x y]
                    v = [x, y, z]

                    result = A * v
                    @test length(result) == 2
                    @test result isa Vector
                    @test result[1] isa ExaModels.AbstractNode
                end

                @testset "Matrix × Matrix (both AbstractNode)" begin
                    result = mat_nodes1 * mat_nodes2
                    @test size(result) == (2, 2)
                    @test result isa Matrix
                    @test result[1, 1] isa ExaModels.AbstractNode
                end

                @testset "Adjoint Vector × Matrix (both AbstractNode)" begin
                    A = [x y; z w; y x]
                    v = [x, y, z]

                    result = v' * A
                    @test size(result) == (1, 2)
                    @test result isa LinearAlgebra.Adjoint
                end

                @testset "Vector + Vector (both AbstractNode)" begin
                    result = vec_nodes1 + vec_nodes2
                    @test length(result) == 3
                    @test result isa Vector
                    @test result[1] isa ExaModels.AbstractNode
                end

                @testset "Vector - Vector (both AbstractNode)" begin
                    result = vec_nodes1 - vec_nodes2
                    @test length(result) == 3
                    @test result isa Vector
                    @test result[1] isa ExaModels.AbstractNode
                end

                @testset "Matrix + Matrix (both AbstractNode)" begin
                    result = mat_nodes1 + mat_nodes2
                    @test size(result) == (2, 2)
                    @test result isa Matrix
                    @test result[1, 1] isa ExaModels.AbstractNode
                end

                @testset "Matrix - Matrix (both AbstractNode)" begin
                    result = mat_nodes1 - mat_nodes2
                    @test size(result) == (2, 2)
                    @test result isa Matrix
                    @test result[1, 1] isa ExaModels.AbstractNode
                end

                @testset "Mixed operations (no standard library conflicts)" begin
                    v_num = [1.0, 2.0, 3.0]
                    v_num_2 = [1.0, 2.0]
                    A_num = [1.0 2.0 3.0; 4.0 5.0 6.0]

                    @test (vec_nodes1 * 2.0) isa Vector
                    @test (2.0 * vec_nodes1) isa Vector
                    @test (mat_nodes1 * 2.0) isa Matrix
                    @test (2.0 * mat_nodes1) isa Matrix
                    @test dot(v_num, vec_nodes1) isa ExaModels.AbstractNode
                    @test dot(vec_nodes1, v_num) isa ExaModels.AbstractNode
                    @test (A_num * vec_nodes1) isa Vector
                    @test (mat_nodes1 * v_num_2) isa Vector
                    @test (A_num * [x y; z w; y x]) isa Matrix
                    @test (mat_nodes1 * mat_nodes2) isa Matrix
                end
            end

            @testset "Canonical nodes" begin
                @testset "zero and one helpers" begin
                    z = zero(ExaModels.AbstractNode)
                    @test z isa ExaModels.Null
                    @test iszero(z.value)
                    @test z.value == 0

                    o = one(ExaModels.AbstractNode)
                    @test o isa ExaModels.Null
                    @test isone(o.value)
                    @test o.value == 1
                end

                @testset "zeros and ones array creation" begin
                    z1 = zeros(ExaModels.AbstractNode, 3)
                    @test length(z1) == 3
                    @test z1 isa Vector{<:ExaModels.AbstractNode}
                    @test all(is_null_zero.(z1))
                    @test all(x -> x isa ExaModels.Null, z1)

                    z2 = zeros(ExaModels.AbstractNode, 2, 3)
                    @test size(z2) == (2, 3)
                    @test z2 isa Matrix{<:ExaModels.AbstractNode}
                    @test all(is_null_zero.(z2))

                    z3 = zeros(ExaModels.AbstractNode, 2, 2, 2)
                    @test size(z3) == (2, 2, 2)
                    @test z3 isa Array{<:ExaModels.AbstractNode, 3}
                    @test all(is_null_zero.(z3))

                    o1 = ones(ExaModels.AbstractNode, 3)
                    @test length(o1) == 3
                    @test o1 isa Vector{<:ExaModels.AbstractNode}
                    @test all(x -> x isa ExaModels.Null && isone(x.value), o1)

                    o2 = ones(ExaModels.AbstractNode, 2, 3)
                    @test size(o2) == (2, 3)
                    @test o2 isa Matrix{<:ExaModels.AbstractNode}
                    @test all(x -> x isa ExaModels.Null && isone(x.value), o2)

                    o3 = ones(ExaModels.AbstractNode, 2, 2, 2)
                    @test size(o3) == (2, 2, 2)
                    @test o3 isa Array{<:ExaModels.AbstractNode, 3}
                    @test all(x -> x isa ExaModels.Null && isone(x.value), o3)

                    z_var = zeros(ExaModels.AbstractNode, 3)
                    @test length(z_var) == 3
                    @test eltype(z_var) <: ExaModels.AbstractNode
                    @test all(is_null_zero.(z_var))

                    o_var = ones(ExaModels.AbstractNode, 2, 2)
                    @test size(o_var) == (2, 2)
                    @test eltype(o_var) <: ExaModels.AbstractNode
                    @test all(x -> x isa ExaModels.Null && isone(x.value), o_var)
                end

                @testset "zeros and ones in operations" begin
                    x, y, z, w = create_nodes()

                    z_vec = zeros(ExaModels.AbstractNode, 3)
                    vec_nodes = [x, y, z]

                    result1 = vec_nodes + z_vec
                    @test result1[1].value == x.value
                    @test result1[2].value == y.value
                    @test result1[3].value == z.value

                    result2 = [ExaModels.Null(2), ExaModels.Null(3), ExaModels.Null(4)] .* z_vec
                    @test all(is_null_zero.(result2))

                    o_vec = ones(ExaModels.AbstractNode, 3)

                    result3 = vec_nodes .* o_vec
                    @test result3[1].value == x.value
                    @test result3[2].value == y.value
                    @test result3[3].value == z.value

                    I_like = diagm(ones(ExaModels.AbstractNode, 2))
                    vec2 = [x, y]
                    result4 = I_like * vec2
                    @test result4[1].value == x.value
                    @test result4[2].value == y.value
                end
            end

            @testset "Zero multiplication optimizations" begin
                x, y, z, w = create_nodes()

                @testset "Scalar × Vector with zero scalar" begin
                    result1 = ExaModels.Null(0) * [1.0, 2.0, 3.0]
                    @test all(is_null_zero.(result1))
                    @test result1 isa Vector{<:ExaModels.AbstractNode}
                    @test length(result1) == 3

                    vec_nodes = [x, y, z]
                    result2 = 0 * vec_nodes
                    @test all(is_null_zero.(result2))
                    @test result2 isa Vector{<:ExaModels.AbstractNode}

                    result3 = 0.0 * vec_nodes
                    @test all(is_null_zero.(result3))
                end

                @testset "Scalar × Vector with zero elements" begin
                    result = x * [0, 1.0, 0, 2.0]
                    @test is_null_zero(result[1])
                    @test !is_null_zero(result[2])
                    @test is_null_zero(result[3])
                    @test !is_null_zero(result[4])
                end

                @testset "Vector × Scalar with zero scalar" begin
                    vec_nodes = [x, y, z]

                    result1 = vec_nodes * 0
                    @test all(is_null_zero.(result1))
                    @test result1 isa Vector{<:ExaModels.AbstractNode}

                    result2 = vec_nodes * 0.0
                    @test all(is_null_zero.(result2))

                    result3 = [1.0, 2.0, 3.0] * ExaModels.Null(0)
                    @test all(is_null_zero.(result3))
                end

                @testset "Vector × Scalar with zero elements" begin
                    result = [0, 1.0, ExaModels.Null(0), 2.0] * x
                    @test is_null_zero(result[1])
                    @test !is_null_zero(result[2])
                    @test is_null_zero(result[3])
                    @test !is_null_zero(result[4])
                end

                @testset "Scalar × Matrix with zero scalar" begin
                    mat_nodes = [x y; z w]

                    result1 = 0 * mat_nodes
                    @test all(is_null_zero.(result1))
                    @test result1 isa Matrix{<:ExaModels.AbstractNode}
                    @test size(result1) == (2, 2)

                    result2 = ExaModels.Null(0) * [1.0 2.0; 3.0 4.0]
                    @test all(is_null_zero.(result2))
                end

                @testset "Matrix × Scalar with zero scalar" begin
                    mat_nodes = [x y; z w]

                    result = mat_nodes * 0.0
                    @test all(is_null_zero.(result))
                    @test result isa Matrix{<:ExaModels.AbstractNode}
                end
            end

            @testset "Zero addition optimizations" begin
                x, y, z, w = create_nodes()

                @testset "Vector + Vector with zero elements" begin
                    vec_nodes = [x, y, z]

                    result1 = vec_nodes + [0, 0, 0]
                    @test result1[1].value == x.value
                    @test result1[2].value == y.value
                    @test result1[3].value == z.value
                    @test result1 isa Vector{<:ExaModels.AbstractNode}

                    result2 = [0.0, 0.0, 0.0] + vec_nodes
                    @test result2[1].value == x.value
                    @test result2[2].value == y.value
                    @test result2[3].value == z.value

                    result3 = vec_nodes + [0, 1.0, 0]
                    @test result3[1].value == x.value
                    @test result3[2].value == y.value + 1.0
                    @test result3[3].value == z.value

                    zero_vec = [ExaModels.Null(0), ExaModels.Null(0), ExaModels.Null(0)]
                    result4 = vec_nodes + zero_vec
                    @test result4[1].value == x.value
                    @test result4[2].value == y.value
                    @test result4[3].value == z.value
                end

                @testset "Matrix + Matrix with zero elements" begin
                    mat_nodes = [x y; z w]

                    result1 = mat_nodes + [0 0; 0 0]
                    @test result1[1, 1].value == x.value
                    @test result1[1, 2].value == y.value
                    @test result1[2, 1].value == z.value
                    @test result1[2, 2].value == w.value
                    @test result1 isa Matrix{<:ExaModels.AbstractNode}

                    result2 = [0.0 0.0; 0.0 0.0] + mat_nodes
                    @test result2[1, 1].value == x.value
                    @test result2[1, 2].value == y.value

                    result3 = mat_nodes + [0 1.0; 0 0]
                    @test result3[1, 1].value == x.value
                    @test result3[1, 2].value == y.value + 1.0
                    @test result3[2, 1].value == z.value
                    @test result3[2, 2].value == w.value
                end
            end

            @testset "Zero subtraction optimizations" begin
                x, y, z, w = create_nodes()

                @testset "Vector - Vector with zero elements" begin
                    vec_nodes = [x, y, z]

                    result1 = vec_nodes - [0, 0, 0]
                    @test result1[1].value == x.value
                    @test result1[2].value == y.value
                    @test result1[3].value == z.value
                    @test result1 isa Vector{<:ExaModels.AbstractNode}

                    result2 = vec_nodes - [0, 1.0, 0]
                    @test result2[1].value == x.value
                    @test result2[2].value == y.value - 1.0
                    @test result2[3].value == z.value

                    zero_vec = [ExaModels.Null(0), ExaModels.Null(0), ExaModels.Null(0)]
                    result3 = vec_nodes - zero_vec
                    @test result3[1].value == x.value
                    @test result3[2].value == y.value
                    @test result3[3].value == z.value
                end

                @testset "Matrix - Matrix with zero elements" begin
                    mat_nodes = [x y; z w]

                    result1 = mat_nodes - [0 0; 0 0]
                    @test result1[1, 1].value == x.value
                    @test result1[1, 2].value == y.value
                    @test result1[2, 1].value == z.value
                    @test result1[2, 2].value == w.value
                    @test result1 isa Matrix{<:ExaModels.AbstractNode}

                    result2 = mat_nodes - [0 1.0; 0 0]
                    @test result2[1, 1].value == x.value
                    @test result2[1, 2].value == y.value - 1.0
                    @test result2[2, 1].value == z.value
                    @test result2[2, 2].value == w.value
                end
            end

            @testset "Scalar operations on Null nodes (+, -, *)" begin
                x, y, z, w = create_nodes()

                e = ExaModels.Node2(+, x, y)
                f = ExaModels.Node2(+, z, w)

                @testset "+ operator rules" begin
                    result1 = ExaModels.Null(3) + ExaModels.Null(5)
                    @test result1 isa ExaModels.Null
                    @test result1.value == 8

                    result2 = ExaModels.Null(3) + e
                    @test result2 isa ExaModels.AbstractNode
                    @test !(result2 isa ExaModels.Null)

                    result3 = e + ExaModels.Null(3)
                    @test result3 isa ExaModels.AbstractNode
                    @test !(result3 isa ExaModels.Null)

                    result4 = e + f
                    @test result4 isa ExaModels.AbstractNode
                end

                @testset "- operator rules" begin
                    result1 = ExaModels.Null(5) - ExaModels.Null(3)
                    @test result1 isa ExaModels.Null
                    @test result1.value == 2

                    result2 = ExaModels.Null(0) - e
                    @test result2 isa ExaModels.Node1

                    result3 = ExaModels.Null(3) - e
                    @test result3 isa ExaModels.AbstractNode
                    @test !(result3 isa ExaModels.Null)

                    result4 = e - ExaModels.Null(3)
                    @test result4 isa ExaModels.AbstractNode
                    @test !(result4 isa ExaModels.Null)

                    result5 = e - f
                    @test result5 isa ExaModels.AbstractNode
                end

                @testset "* operator rules" begin
                    result1 = ExaModels.Null(3) * ExaModels.Null(5)
                    @test result1 isa ExaModels.Null
                    @test result1.value == 15

                    result2 = ExaModels.Null(0) * e
                    @test result2 isa ExaModels.Null
                    @test iszero(result2.value)

                    result3 = e * ExaModels.Null(0)
                    @test result3 isa ExaModels.Null
                    @test iszero(result3.value)

                    result4 = ExaModels.Null(3) * e
                    @test result4 isa ExaModels.AbstractNode
                    @test !(result4 isa ExaModels.Null)

                    result5 = e * ExaModels.Null(3)
                    @test result5 isa ExaModels.AbstractNode
                    @test !(result5 isa ExaModels.Null)

                    result6 = e * f
                    @test result6 isa ExaModels.AbstractNode
                end
            end

            @testset "sum function" begin
                x, y, z, w = create_nodes()

                @testset "sum with zeros" begin
                    result1 = sum([zero(ExaModels.AbstractNode), zero(ExaModels.AbstractNode), zero(ExaModels.AbstractNode)])
                    @test result1 isa ExaModels.Null
                    @test iszero(result1.value)

                    result2 = sum([zero(ExaModels.AbstractNode), zero(ExaModels.AbstractNode), x, zero(ExaModels.AbstractNode)])
                    @test result2 isa ExaModels.Null
                    @test result2.value == x.value

                    result3 = sum([zero(ExaModels.AbstractNode), x, zero(ExaModels.AbstractNode), y, zero(ExaModels.AbstractNode)])
                    @test result3 isa ExaModels.Null
                    @test result3.value == x.value + y.value
                end

                @testset "sum with single element" begin
                    result1 = sum([x])
                    @test result1 isa ExaModels.Null
                    @test result1.value == x.value

                    result2 = sum([zero(ExaModels.AbstractNode)])
                    @test result2 isa ExaModels.Null
                    @test iszero(result2.value)
                end

                @testset "sum with all non-zeros" begin
                    result1 = sum([x, y, z])
                    @test result1 isa ExaModels.Null
                    @test result1.value == x.value + y.value + z.value
                end

                @testset "sum on matrices" begin
                    mat = [x zero(ExaModels.AbstractNode); zero(ExaModels.AbstractNode) y]
                    result = sum(mat)
                    @test result isa ExaModels.Null
                    @test result.value == x.value + y.value
                end
            end

            @testset "Optimized dot product" begin
                x, y, z, t = create_nodes()

                @testset "dot([1, 0, 1, 0], [x, y, z, t]) = x + z" begin
                    result = dot([1, 0, 1, 0], [x, y, z, t])
                    @test result isa ExaModels.Null
                    @test result.value == x.value + z.value
                end

                @testset "dot with all zeros" begin
                    result = dot([0, 0, 0], [x, y, z])
                    @test result isa ExaModels.Null
                    @test iszero(result.value)
                end

                @testset "dot with all ones" begin
                    result = dot([1, 1, 1], [x, y, z])
                    @test result isa ExaModels.Null
                    @test result.value == x.value + y.value + z.value
                end

                @testset "dot with single non-zero" begin
                    result = dot([0, 1, 0], [x, y, z])
                    @test result isa ExaModels.Null
                    @test result.value == y.value
                end
            end

            @testset "Matrix operations with optimization" begin
                x, y, z, w = create_nodes()

                @testset "Identity matrix multiplication" begin
                    I2 = [1 0; 0 1]
                    vec = [x, y]
                    result = I2 * vec

                    @test result[1] isa ExaModels.Null
                    @test result[1].value == x.value
                    @test result[2] isa ExaModels.Null
                    @test result[2].value == y.value
                end

                @testset "Zero matrix multiplication" begin
                    Z2 = [0 0; 0 0]
                    vec = [x, y]
                    result = Z2 * vec

                    @test result[1] isa ExaModels.Null
                    @test iszero(result[1].value)
                    @test result[2] isa ExaModels.Null
                    @test iszero(result[2].value)
                end

                @testset "Sparse matrix multiplication" begin
                    A = [1 0 0; 0 0 1]
                    vec = [x, y, z]
                    result = A * vec

                    @test result[1] isa ExaModels.Null
                    @test result[1].value == x.value
                    @test result[2] isa ExaModels.Null
                    @test result[2].value == z.value
                end
            end

            @testset "SubArray, ReshapedArray, ReinterpretArray" begin
                x, y, z, w = create_nodes()

                @testset "SubArray (views) - vectors of Real" begin
                    v_num = [1.0, 2.0, 3.0, 4.0]
                    vec_nodes = [x, y, z, w]

                    v_view = @view v_num[1:3]
                    @test v_view isa SubArray

                    result1 = dot(v_view, vec_nodes[1:3])
                    @test result1 isa ExaModels.AbstractNode

                    result2 = dot(vec_nodes[1:3], v_view)
                    @test result2 isa ExaModels.AbstractNode

                    result3 = x * v_view
                    @test length(result3) == 3
                    @test result3[1] isa ExaModels.AbstractNode

                    result4 = 2.0 * vec_nodes[1:3]
                    @test length(result4) == 3

                    result5 = v_view * x
                    @test length(result5) == 3
                    @test result5[1] isa ExaModels.AbstractNode

                    result6 = vec_nodes[1:3] + v_view
                    @test length(result6) == 3
                    @test result6[1] isa ExaModels.AbstractNode

                    result7 = v_view + vec_nodes[1:3]
                    @test length(result7) == 3

                    result8 = vec_nodes[1:3] - v_view
                    @test length(result8) == 3
                    @test result8[1] isa ExaModels.AbstractNode

                    result9 = v_view - vec_nodes[1:3]
                    @test length(result9) == 3

                    result10 = norm(@view vec_nodes[1:3])
                    @test result10 isa ExaModels.AbstractNode
                end

                @testset "SubArray (views) - vectors of AbstractNode" begin
                    vec_nodes = [x, y, z, w]

                    v_view = @view vec_nodes[1:3]
                    @test v_view isa SubArray

                    v_view2 = @view vec_nodes[2:4]
                    result1 = v_view + v_view2
                    @test length(result1) == 3
                    @test result1[1] isa ExaModels.AbstractNode

                    result2 = v_view - v_view2
                    @test length(result2) == 3

                    result3 = dot(v_view, v_view2)
                    @test result3 isa ExaModels.AbstractNode

                    result4 = 2.5 * v_view
                    @test length(result4) == 3
                    @test result4[1] isa ExaModels.AbstractNode

                    result5 = x * v_view
                    @test length(result5) == 3
                end

                @testset "SubArray (views) - matrices of Real" begin
                    A_num = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
                    mat_nodes = [x y z; w x y; z w x]

                    A_view = @view A_num[1:2, 1:2]
                    @test A_view isa SubArray
                    @test size(A_view) == (2, 2)

                    result1 = x * A_view
                    @test size(result1) == (2, 2)
                    @test result1[1, 1] isa ExaModels.AbstractNode

                    result2 = 3.0 * (@view mat_nodes[1:2, 1:2])
                    @test size(result2) == (2, 2)

                    result3 = A_view * x
                    @test size(result3) == (2, 2)
                    @test result3[1, 1] isa ExaModels.AbstractNode

                    vec_nodes = [x, y]
                    result4 = A_view * vec_nodes
                    @test length(result4) == 2
                    @test result4[1] isa ExaModels.AbstractNode

                    result5 = (@view mat_nodes[1:2, 1:2]) + A_view
                    @test size(result5) == (2, 2)
                    @test result5[1, 1] isa ExaModels.AbstractNode
                end

                @testset "SubArray (views) - matrices of AbstractNode" begin
                    mat_nodes = [x y z; w x y; z w x]

                    A_view = @view mat_nodes[1:2, 1:2]
                    @test A_view isa SubArray
                    @test size(A_view) == (2, 2)

                    B_view = @view mat_nodes[2:3, 2:3]
                    result1 = A_view * B_view
                    @test size(result1) == (2, 2)
                    @test result1[1, 1] isa ExaModels.AbstractNode

                    result2 = A_view + B_view
                    @test size(result2) == (2, 2)

                    result3 = A_view - B_view
                    @test size(result3) == (2, 2)

                    result4 = adjoint(A_view)
                    @test size(result4) == (2, 2)

                    result5 = transpose(A_view)
                    @test size(result5) == (2, 2)
                end

                @testset "ReshapedArray - vectors to matrices" begin
                    v_num = [1.0, 2.0, 3.0, 4.0]
                    vec_nodes = [x, y, z, w]

                    A_reshaped = reshape(v_num, 2, 2)
                    @test A_reshaped isa Union{Matrix, Base.ReshapedArray}

                    result1 = x * A_reshaped
                    @test size(result1) == (2, 2)
                    @test result1[1, 1] isa ExaModels.AbstractNode

                    mat_reshaped = reshape(vec_nodes, 2, 2)
                    @test mat_reshaped isa Union{Matrix, Base.ReshapedArray}

                    result2 = 2.0 * mat_reshaped
                    @test size(result2) == (2, 2)
                    @test result2[1, 1] isa ExaModels.AbstractNode

                    result3 = det(mat_reshaped)
                    @test result3 isa ExaModels.AbstractNode

                    result4 = tr(mat_reshaped)
                    @test result4 isa ExaModels.AbstractNode
                end

                @testset "ReshapedArray - matrices to vectors" begin
                    A_num = [1.0 2.0; 3.0 4.0]
                    mat_nodes = [x y; z w]

                    v_reshaped = reshape(A_num, 4)
                    @test v_reshaped isa Union{Vector, Base.ReshapedArray}

                    result1 = x * v_reshaped
                    @test length(result1) == 4
                    @test result1[1] isa ExaModels.AbstractNode

                    vec_reshaped = reshape(mat_nodes, 4)
                    @test vec_reshaped isa Union{Vector, Base.ReshapedArray}

                    result2 = 2.0 * vec_reshaped
                    @test length(result2) == 4
                    @test result2[1] isa ExaModels.AbstractNode

                    result3 = norm(vec_reshaped)
                    @test result3 isa ExaModels.AbstractNode

                    result4 = dot(v_reshaped, vec_reshaped)
                    @test result4 isa ExaModels.AbstractNode
                end

                @testset "ReinterpretArray - Complex to Real" begin
                    complex_vec = ComplexF64[1.0 + 2.0im, 3.0 + 4.0im, 5.0 + 6.0im]

                    real_reinterp = reinterpret(Float64, complex_vec)
                    @test real_reinterp isa Base.ReinterpretArray
                    @test length(real_reinterp) == 6

                    vec_nodes = [x, y, z, w, ExaModels.Null(5), ExaModels.Null(6)]

                    result1 = dot(real_reinterp, vec_nodes)
                    @test result1 isa ExaModels.AbstractNode

                    result2 = dot(vec_nodes, real_reinterp)
                    @test result2 isa ExaModels.AbstractNode

                    result3 = x * real_reinterp
                    @test length(result3) == 6
                    @test result3[1] isa ExaModels.AbstractNode

                    result4 = real_reinterp * x
                    @test length(result4) == 6
                    @test result4[1] isa ExaModels.AbstractNode

                    result5 = vec_nodes + real_reinterp
                    @test length(result5) == 6
                    @test result5[1] isa ExaModels.AbstractNode

                    result6 = real_reinterp + vec_nodes
                    @test length(result6) == 6

                    result7 = vec_nodes - real_reinterp
                    @test length(result7) == 6
                    @test result7[1] isa ExaModels.AbstractNode

                    result8 = real_reinterp - vec_nodes
                    @test length(result8) == 6
                end

                @testset "Mixed wrapper combinations" begin
                    v_num = [1.0, 2.0, 3.0, 4.0]
                    vec_nodes = [x, y, z, w]

                    v_view = @view v_num[1:3]
                    v_reshaped = reshape([x, y, z], 3)
                    result1 = v_view + v_reshaped
                    @test length(result1) == 3
                    @test result1[1] isa ExaModels.AbstractNode

                    A_num = [1.0 2.0 3.0; 4.0 5.0 6.0; 7.0 8.0 9.0]
                    A_view = @view A_num[1:3, 1:3]
                    vec_reshaped = reshape([x, y, z], 3)
                    result2 = A_view * vec_reshaped
                    @test length(result2) == 3
                    @test result2[1] isa ExaModels.AbstractNode

                    complex_vec = ComplexF64[1.0 + 2.0im, 3.0 + 4.0im]
                    real_reinterp = reinterpret(Float64, complex_vec)
                    vec_view = @view vec_nodes[1:4]
                    result3 = real_reinterp + vec_view
                    @test length(result3) == 4
                    @test result3[1] isa ExaModels.AbstractNode
                end

                @testset "diagm with views and reshaped arrays" begin
                    vec_nodes = [x, y, z]

                    v_view = @view vec_nodes[1:2]
                    result1 = diagm(v_view)
                    @test size(result1) == (2, 2)
                    @test result1[1, 1] isa ExaModels.AbstractNode
                    @test is_null_zero(result1[1, 2])

                    v_reshaped = reshape([x, y], 2)
                    result2 = diagm(v_reshaped)
                    @test size(result2) == (2, 2)
                    @test result2[1, 1] isa ExaModels.AbstractNode

                    result3 = diagm(1 => v_view)
                    @test size(result3) == (3, 3)
                    @test result3[1, 2] isa ExaModels.AbstractNode
                end

                @testset "diag with views and reshaped arrays" begin
                    mat_nodes = [x y z; w x y; z w x]

                    A_view = @view mat_nodes[1:2, 1:2]
                    result1 = diag(A_view)
                    @test length(result1) == 2
                    @test result1[1] isa ExaModels.AbstractNode

                    vec = [x, y, z, w]
                    A_reshaped = reshape(vec, 2, 2)
                    result2 = diag(A_reshaped)
                    @test length(result2) == 2
                    @test result2[1] isa ExaModels.AbstractNode
                end

                @testset "Adjoint operations with views" begin
                    vec_nodes = [x, y, z]
                    A_num = [1.0 2.0; 3.0 4.0; 5.0 6.0]

                    v_view = @view vec_nodes[1:3]
                    v_num = [1.0, 2.0, 3.0]
                    result0 = v_view' * v_num
                    @test result0 isa ExaModels.AbstractNode

                    result0b = v_num' * v_view
                    @test result0b isa ExaModels.AbstractNode

                    v_view2 = @view vec_nodes[1:3]
                    result0c = v_view' * v_view2
                    @test result0c isa ExaModels.AbstractNode

                    result1 = v_view' * A_num
                    @test size(result1) == (1, 2)
                    @test result1 isa LinearAlgebra.Adjoint

                    mat_nodes = [x y z; w x y; z w x]
                    A_view = @view mat_nodes[1:2, 1:2]
                    v_num2 = [1.0, 2.0]
                    result2 = v_num2' * A_view
                    @test size(result2) == (1, 2)

                    vec_reshaped = reshape([x, y, z], 3)
                    result3 = vec_reshaped' * v_num
                    @test result3 isa ExaModels.AbstractNode

                    result4 = v_num' * vec_reshaped
                    @test result4 isa ExaModels.AbstractNode
                end
            end
        end
    end
end

end # module
