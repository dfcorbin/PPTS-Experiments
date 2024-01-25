@testset "GaussianEnv" begin
    mean_fn(x::Vector{Float64}, a::Int64) = a == 1 ? x[1] : -x[1]
    env = GaussianEnv(mean_fn, 1, 2; noise = 0.0, lower = -1.0, upper = 1.0)
    @test isa(get_context(env), Vector{Float64})
    @test get_num_dims(env) == 1
    @test get_num_acts(env) == 2
    env.ctex[1] = 1.0
    @test act!(env, 1) == (1.0, 0.0)
    env.ctex[1] = 1.0
    @test act!(env, 2) == (-1.0, 2.0)
    @test env.ctex != [1.0]
end

@testset "WheelEnv" begin
    env = WheelEnv((0.2, 0.1, 0.6), 0.0, 0.5)

    # Check inner circle behaviour.
    env.ctex = [0.0, 0.0]
    @test act!(env, 1) == (0.2, 0.0)
    for a = 2:5
        env.ctex = [0.0, 0.0]
        @test act!(env, a) == (0.1, 0.1)
    end

    # Test outer circle behaviour.
    outer_ctexs = [[-0.6, 0.6], [0.6, 0.6], [0.6, -0.6], [-0.6, -0.6]]
    outer_optimal = [4, 2, 3, 5]
    for a = 2:5
        for (c, ctex) in enumerate(outer_ctexs)
            env.ctex = ctex
            reward, regret = act!(env, a)
            if !(outer_optimal[c] == a ? reward == 0.6 : reward == 0.1)
                println(
                    "outer_optimal = ",
                    outer_optimal[a-1],
                    ", a = ",
                    a,
                    ", reward = ",
                    reward
                )
            end
            @test outer_optimal[c] == a ? reward ≈ 0.6 : reward ≈ 0.1
            @test outer_optimal[c] == a ? regret ≈ 0.0 : regret ≈ 0.5
        end
    end
    env.ctex = [0.6, 0.0]
    reward, regret = act!(env, 1)
    @test reward ≈ 0.2
    @test regret ≈ 0.4
end

@testset "LatentGaussianEnv" begin
    tform(ctex) = [ctex[1] + 1.0]
    mean_fn(x::Vector{Float64}, a::Int64) = a == 1 ? x[1] : -x[1]
    env = LatentGaussianEnv(tform, mean_fn, 2, 2; noise = 0.0, lower = -1.0, upper = 1.0)
    @test get_context(env) == tform(get_context(env.latent_env))
    @test get_num_dims(env) == 1
    @test get_num_acts(env) == 2
    latent_ctex = get_context(env.latent_env)
    reward, regret = act!(env, 1)
    @test reward ≈ latent_ctex[1]
    latent_ctex = get_context(env.latent_env)
    reward, regret = act!(env, 2)
    @test reward ≈ -latent_ctex[1]
end

@testset "LabelledDataEnv" begin
    ctexs = [
        1.0 0.0
        0.0 1.0
    ]
    labels = [1, 2]
    env = LabelledDataEnv(ctexs, labels)
    @test get_num_dims(env) == 2
    @test get_num_acts(env) == 2

    env.ctexs = ctexs
    env.labels = labels
    reward, regret = act!(env, 1)
    @test reward == 1.0
    @test regret == 0.0
    @test get_context(env) == ctexs[:, 2]

    env = LabelledDataEnv(ctexs, labels)
    env.ctexs = ctexs
    env.labels = labels
    reward, regret = act!(env, 2)
    @test reward == 0.0
    @test regret == 1.0
end