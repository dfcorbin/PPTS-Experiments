@testset "Sample Gaussian data" begin
    mean_fn = x -> 100.0
    inputs, targets = gaussian_data(mean_fn, 2, 5; noise = 0.1, lower = -0.1, upper = 0.1)
    @test typeof(inputs) == Array{Float64,2}
    @test typeof(targets) == Array{Float64,1}
    @test size(inputs) == (2, 5)
    @test size(targets) == (5,)
    @test all(-0.1 .<= inputs .<= 0.1)
    @test all(90 .<= targets .<= 110)
end