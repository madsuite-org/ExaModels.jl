module GetterSetterTest

using Test, ExaModels

function build_model()
    c = ExaCore(concrete = Val(true))
    c, x = add_var(c, 3; start = 1.0, lvar = -2.0, uvar = 5.0)
    c, y = add_var(c, 2; start = 0.5, lvar = 0.0, uvar = 1.0)
    c, θ1 = add_par(c, [10.0, 20.0, 30.0])
    c, θ2 = add_par(c, 2; value = 7.0)
    c, g = add_con(c, x[i] + x[i + 1] for i in 1:2; lcon = -1.0, ucon = 1.0, start = 0.1)
    c, h = add_con(c, y[1] - y[2]; lcon = 0.0, ucon = 0.0, start = 0.5)
    return ExaModel(c), x, y, θ1, θ2, g, h
end

function runtests()
    return @testset "Getter/setter API" begin

        m, x, y, θ1, θ2, g, h = build_model()

        @testset "get_value — parameters" begin
            @test get_value(m, θ1) == [10.0, 20.0, 30.0]
            @test get_value(m, θ2) == [7.0, 7.0]
        end

        @testset "set_value! — parameters" begin
            set_value!(m, θ1, [1.0, 2.0, 3.0])
            @test get_value(m, θ1) == [1.0, 2.0, 3.0]

            set_value!(m, θ2, [9.0, 8.0])
            @test get_value(m, θ2) == [9.0, 8.0]

            @test_throws DimensionMismatch set_value!(m, θ1, [1.0, 2.0])
            @test_throws DimensionMismatch set_value!(m, θ2, [1.0])
        end

        @testset "get_start / get_lvar / get_uvar — variables" begin
            @test get_start(m, x) == [1.0, 1.0, 1.0]
            @test get_lvar(m, x) == [-2.0, -2.0, -2.0]
            @test get_uvar(m, x) == [5.0, 5.0, 5.0]
            @test get_start(m, y) == [0.5, 0.5]
            @test get_lvar(m, y) == [0.0, 0.0]
            @test get_uvar(m, y) == [1.0, 1.0]
        end

        @testset "set_start! / set_lvar! / set_uvar! — variables" begin
            set_start!(m, x, [2.0, 3.0, 4.0])
            @test get_start(m, x) == [2.0, 3.0, 4.0]

            set_lvar!(m, x, [-5.0, -6.0, -7.0])
            @test get_lvar(m, x) == [-5.0, -6.0, -7.0]

            set_uvar!(m, x, [10.0, 11.0, 12.0])
            @test get_uvar(m, x) == [10.0, 11.0, 12.0]

            @test_throws DimensionMismatch set_start!(m, x, [1.0])
            @test_throws DimensionMismatch set_lvar!(m, x, [1.0])
            @test_throws DimensionMismatch set_uvar!(m, x, [1.0])
        end

        @testset "get_start / get_lcon / get_ucon — constraints" begin
            @test get_start(m, g) == [0.1, 0.1]
            @test get_lcon(m, g) == [-1.0, -1.0]
            @test get_ucon(m, g) == [1.0, 1.0]
            @test get_start(m, h) == [0.5]
            @test get_lcon(m, h) == [0.0]
            @test get_ucon(m, h) == [0.0]
        end

        @testset "set_start! / set_lcon! / set_ucon! — constraints" begin
            set_start!(m, g, [0.9, 0.8])
            @test get_start(m, g) == [0.9, 0.8]

            set_lcon!(m, g, [-3.0, -4.0])
            @test get_lcon(m, g) == [-3.0, -4.0]

            set_ucon!(m, g, [3.0, 4.0])
            @test get_ucon(m, g) == [3.0, 4.0]

            set_start!(m, h, [0.2])
            @test get_start(m, h) == [0.2]

            @test_throws DimensionMismatch set_start!(m, g, [1.0])
            @test_throws DimensionMismatch set_lcon!(m, g, [1.0])
            @test_throws DimensionMismatch set_ucon!(m, g, [1.0])
        end

        @testset "get_value returns a view (mutations visible)" begin
            v = get_value(m, θ1)
            v[1] = 99.0
            @test get_value(m, θ1)[1] == 99.0
        end

    end
end

end # module GetterSetterTest
