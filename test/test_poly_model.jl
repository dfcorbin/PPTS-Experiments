@testset "PolyModel" begin
    num_obs = 1000
    num_dims = 2
    space = HyperRectangle(num_dims; lower = -1.0, upper = 1.0)
    degree = 2
    basis = trunc_tensor_product_basis(num_dims, degree)
    num_bfuns = length(basis)
    true_coefs = 10 * randn(num_bfuns)

    # Test batch fit.
    f = x -> true_coefs' * Bandits.legendre_expansion(x, basis, space)
    inputs, targets = gaussian_data(f, num_dims, num_obs; noise = 0.01)
    model = PolyModel(basis, space)
    fit!(model, inputs, targets)
    @test isapprox(get_coefs(model), true_coefs, rtol = 0.05)
    true_means = Bandits.legendre_expansion(inputs, basis, space)' * true_coefs
    @test isapprox(predict(model, inputs), true_means, rtol = 0.05)

    # Test sequential fit.
    f = x -> true_coefs' * Bandits.legendre_expansion(x, basis, space)
    inputs, targets = gaussian_data(f, num_dims, num_obs; noise = 0.01)
    model = PolyModel(basis, space)
    for (inp, trg) in zip(eachcol(inputs), targets)
        fit!(model, inp, trg)
    end
    @test isapprox(get_coefs(model), true_coefs, rtol = 0.05)
    true_mean = Bandits.legendre_expansion(inputs[:, 1], basis, space)' * true_coefs
    @test isapprox(predict(model, inputs[:, 1]), true_mean, rtol = 0.05)
end