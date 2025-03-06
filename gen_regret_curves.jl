using Pkg: Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

include("src/Bandits.jl")
using .Bandits
using DataFrames: DataFrame
using Plots
import CSV
import Distributions as Dst

function sample_non_linear_env()
    num_dims = 5
    num_acts = 5

    outer_coefs = [rand(Dst.Normal(0, 5), num_dims) for _ = 1:num_acts]
    sin_coefs = [rand(Dst.Gamma(π, 1), num_dims) for _ = 1:num_acts]
    cos_coefs = [rand(Dst.Gamma(π, 1), num_dims) for _ = 1:num_acts]

    mean_fn = function (ctex::Vector{Float64}, action::Int64)
        mean = 0.0
        for dim = 1:(num_dims-1)
            mean +=
                outer_coefs[action][dim] *
                sin(sin_coefs[action][dim] * ctex[dim]) *
                cos(cos_coefs[action][dim+1] * ctex[dim+1])
        end
        return mean
    end

    return GaussianEnv(
        mean_fn,
        num_dims,
        num_acts,
        noise = 1.0,
        lower = -1.0,
        upper = 1.0
    )
end

function sample_near_linear_env()
    num_dims = 5
    num_acts = 5

    outer_coefs = [rand(Dst.Normal(0, 5), num_dims) for _ = 1:num_acts]
    sin_coefs = [rand(Dst.Uniform(0, 2 * π), num_dims) for _ = 1:num_acts]

    mean_fn = function (ctex::Vector{Float64}, action::Int64)
        mean = 0.0
        for dim = 1:num_dims
            mean +=
                outer_coefs[action][dim] * sin(sin_coefs[action][dim] + π * ctex[dim] / 4)
        end
        return mean
    end

    return GaussianEnv(
        mean_fn,
        num_dims,
        num_acts,
        noise = 1.0,
        lower = -1.0,
        upper = 1.0
    )
end

function sample_latent_linear_env()
    num_latent_dims = 5
    num_acts = 5
    coefs = [rand(Dst.Normal(0, 5), num_latent_dims) for _ = 1:num_acts]

    mean_fn = function (context::Vector{Float64}, action::Int64)
        return coefs[action]' * context
    end

    tform = function (context::Vector{Float64})
        return [
            context[1] * exp(context[2]),
            context[2] / (2 + context[1]),
            context[3] * context[4] / (3 + context[5]),
            log(1 + exp(context[3])),
            exp(context[4] * context[5])
        ]
    end

    return LatentGaussianEnv(
        tform,
        mean_fn,
        num_latent_dims,
        num_acts;
        noise = 1.0,
        lower = -1.0,
        upper = 1.0
    )
end

function sample_shuttle_env()
    data = CSV.read("Data/shuttle.csv", DataFrame; header = 0)
    contexts = Matrix{Float64}(Matrix{Float64}(data[:, 1:9])')
    labels = Vector{Int64}(data[:, end])
    return LabelledDataEnv(contexts, labels)
end

sample_wheel07_env() = WheelEnv((1.2, 1.0, 50.0), 1.0, 0.7)
sample_wheel09_env() = WheelEnv((1.2, 1.0, 50.0), 1.0, 0.9)

# Define global simulation parameters.
num_steps = 20000
initial_steps = 20
num_retrain_steps = 30
rate = (num_steps / initial_steps)^(1 / num_retrain_steps)
retrain_steps = [initial_steps * rate^i for i = 0:(num_retrain_steps-1)]
retrain_steps = stack(n -> ceil(Int64, n), retrain_steps)
inflation = 10.0
penalty = 0.01
prior_shape = 0.01
prior_scale = 0.01
min_data_hard = 2
min_data_ratio = 1.0

# Missing shuttle environment
for env_gen in [sample_shuttle_env]
# for env_gen in [sample_non_linear_env, sample_near_linear_env, sample_latent_linear_env, sample_wheel07_env, sample_wheel09_env]
    plt = plot(xlab="t", ylab="Total Regret")
    for i in 1:10
        println("Simuation $i/10")
        env = env_gen()
        pol = PartitionedPolyTS(
            get_num_dims(env),
            get_num_acts(env),
            initial_steps,
            retrain_steps;
            inflation=inflation,
            max_models=200,
            max_degree=5,
            num_bfuns=15,
            min_data_hard=min_data_hard,
            min_data_ratio=min_data_ratio,
            penalty=penalty,
            prior_shape=prior_shape,
            prior_scale=prior_scale
        )
        polcp = deepcopy(pol)
        regret = run!(env, polcp, 20000, verbose=true)
        plot!(plt, cumsum(regret), label=nothing, c=:black)
    end
    savefig(plt, "figures/$(env_gen)")
end