@testset "LinearModel" begin
    num_obs = 1000
    num_dims = 2
    true_coefs = 10 * randn(num_dims)

    # Test batch fit.
    inputs, targets = gaussian_data(x -> true_coefs' * x, num_dims, num_obs; noise = 0.01)
    model = LinearModel(num_dims)
    fit!(model, inputs, targets)
    @test isapprox(get_coefs(model), true_coefs, rtol = 0.05)
    @test isapprox(predict(model, inputs), inputs' * true_coefs, rtol = 0.05)

    # Test sequential fit.
    inputs, targets = gaussian_data(x -> true_coefs' * x, num_dims, num_obs; noise = 0.01)
    model = LinearModel(num_dims)
    for (inp, trg) in zip(eachcol(inputs), targets)
        fit!(model, inp, trg)
    end
    @test isapprox(get_coefs(model), true_coefs, rtol = 0.05)
    @test isapprox(predict(model, inputs[:, 1]), inputs[:, 1]' * true_coefs, rtol = 0.05)
    return nothing
end