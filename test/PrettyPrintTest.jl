module PrettyPrintTest

using Test, ExaModels

# short type printing is opt-in
ExaModels.enable_short_type_printing()

function runtests()
    @testset "Pretty printing" begin
        c = ExaCore(concrete = Val(true))
        c, x = add_var(c, 5)

        @testset "Node expression strings" begin
            @test sprint(show, sin(x[1]))      == "sin(x[1])"
            @test sprint(show, x[1] + x[2])    == "x[1] + x[2]"
            @test sprint(show, x[1] * x[2])    == "x[1] * x[2]"
            @test sprint(show, x[1]^2)           == "x[1]^2"  # abs2 specialization
            @test sprint(show, x[1] - x[2])     == "x[1] - x[2]"
            @test sprint(show, x[1] / x[2])     == "x[1] / x[2]"
        end

        @testset "Identity simplification" begin
            n = ExaModels.Node2(+, ExaModels.Var(1), ExaModels.Null(nothing))
            @test sprint(show, n) == "x[1]"
            n2 = ExaModels.Node2(*, ExaModels.Var(1), ExaModels.Null(1))
            @test sprint(show, n2) == "x[1]"
            n3 = ExaModels.Node2(^, ExaModels.Var(1), ExaModels.Null(1))
            @test sprint(show, n3) == "x[1]"
        end

        @testset "Iteration variable" begin
            ps = ExaModels.DataSource()
            @test sprint(show, ps) == "i"
            pi = ExaModels.DataIndexed(ps, :cost)
            @test sprint(show, pi) == "i.cost"
        end

        @testset "Named variable" begin
            c2 = ExaCore(concrete = Val(true))
            c2, z = add_var(c2, 4; name=Val(:z))
            @test occursin("z ∈ R^{4}", sprint(show, z))
        end

        @testset "Short type display" begin
            @test sprint(show, typeof(sin(x[1])))      == "Node1{sin,…}"
            @test sprint(show, typeof(x[1] + x[2]))     == "Node2{+,…}"
            @test sprint(show, typeof(ExaModels.Var(1))) == "Var{…}"
        end

        @testset "fulltype" begin
            s = sprint(ExaModels.fulltype, sin(x[1]))
            @test occursin("typeof(sin)", s)
            @test occursin("Var", s)
        end

        @testset "text/plain display" begin
            node = sin(x[1]) + x[2]
            s = sprint(show, MIME"text/plain"(), node)
            @test occursin("Node2{+}", s)
            @test occursin("sin(x[1])", s)
        end
        
        @testset "Expression show" begin
            c3 = ExaCore(concrete = Val(true))
            c3, y = add_var(c3, 3)
            c3, expr = add_expr(c3, y[i]^2 for i in 1:3)
            s = sprint(show, expr)
            @test occursin("Subexpression", s)
            @test occursin("x[i]^2", s)
        end
    end
end

end # module
