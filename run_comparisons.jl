env_names =
    ["non_linear", "near_linear", "latent_linear", "shuttle", "wheel07", "wheel09"]
(length(ARGS) != 1) && error("Must pass one of $env_names as as argument.")
!(ARGS[1] in env_names) && error("Invalid environment name. Choose from: $env_names.")
env_name = ARGS[1]

using Pkg: Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

include("src/Bandits.jl")
using .Bandits
using DataFrames: DataFrame
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

print("\n\nRunning simulations for: $(ARGS[1])...\n\n")

# Define global simulation parameters.
num_sims = 100
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

# Set up policies.
num_dims = get_num_dims(getfield(Main, Symbol("sample_$(env_name)_env"))())
num_acts = get_num_acts(getfield(Main, Symbol("sample_$(env_name)_env"))())
policies = Dict()
policies["random"] = RandomPolicy(num_acts)
policies["neighbor_ucb"] = NeighborUCB(
    num_dims,
    num_acts,
    initial_steps,
    retrain_steps;
    theta = 1.0,
    varphi = 1.0
)
policies["partitioned_poly_thompson"] = PartitionedPolyTS(
    num_dims,
    num_acts,
    initial_steps,
    retrain_steps;
    inflation = inflation,
    max_models = 200,
    max_degree = 5,
    num_bfuns = 15,
    min_data_hard = min_data_hard,
    min_data_ratio = min_data_ratio,
    penalty = penalty,
    prior_shape = prior_shape,
    prior_scale = prior_scale
)
policies["poly_thompson"] = PartitionedPolyTS(
    num_dims,
    num_acts,
    initial_steps,
    retrain_steps;
    inflation = inflation,
    max_models = 1,
    max_degree = 5,
    num_bfuns = 100,
    min_data_hard = min_data_hard,
    min_data_ratio = min_data_ratio,
    penalty = penalty,
    prior_shape = prior_shape,
    prior_scale = prior_scale
)
policies["partitioned_linear_thompson"] = PartitionedPolyTS(
    num_dims,
    num_acts,
    initial_steps,
    retrain_steps;
    inflation = inflation,
    max_models = 200,
    max_degree = 1,
    num_bfuns = 100,
    min_data_hard = min_data_hard,
    min_data_ratio = min_data_ratio,
    penalty = penalty,
    prior_shape = prior_shape,
    prior_scale = prior_scale
)
policies["neural_linear"] = NeuralLinear(
    num_dims,
    num_acts,
    initial_steps,
    retrain_steps;
    inflation = 1.0, # Doesn't benefit from inflation
    widths = [100, 100],
    num_epochs = 50,
    batch_size = 32,
    penalty = penalty,
    prior_shape = prior_shape,
    prior_scale = prior_scale
)

try
    mkdir("Results")
    print("\nCreated Results directory...\n")
catch
    print("\nAssuming Results directory already exists...\n")
end

for sim = 1:num_sims
    env = getfield(Main, Symbol("sample_$(env_name)_env"))() # Sample new random environment.
    pol_regrets = Dict{String,Vector{Float64}}()
    for pol_key in keys(policies)
        print("\nSimulation: $sim/$num_sims | Policy: $pol_key\n")
        pol_regrets[pol_key] =
            run!(deepcopy(env), deepcopy(policies[pol_key]), num_steps; verbose = true)
    end

    # We only save the regrets once all policies have been tested on the same environment.
    for (pol_key, regret) in pol_regrets
        file_name = env_name * "-" * pol_key * ".csv"

        CSV.write("Results/" * file_name, DataFrame(regret', :auto); append = true)
    end
end
