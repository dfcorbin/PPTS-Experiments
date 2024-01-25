using Test: @test, @testset

include("../src/Bandits.jl")
using .Bandits
import .Bandits

println("Running PPTS-Experiments tests...\n")

@testset "Bandits" begin
    @testset "Utilities" include("test_utils.jl")
    @testset "CMAB API" include("test_api.jl")
    @testset "Environments" include("test_envs.jl")
    include("test_linear_model.jl")
    include("test_poly_model.jl")
end

println("\nSuccessfully finished PPTS-Experiments tests.\n")