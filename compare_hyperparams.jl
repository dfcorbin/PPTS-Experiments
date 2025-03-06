using Pkg: Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

include("src/Bandits.jl")
using .Bandits
using Plots: contourf, savefig
using Statistics: mean, std
import Distributions as Dst
import DataFrames as Dfs

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

# Define global simulation parameters.
num_sims = 20
num_steps = 10000
initial_steps = 20
num_retrain_steps = 15
rate = (num_steps / initial_steps)^(1 / num_retrain_steps)
retrain_steps = [initial_steps * rate^i for i = 0:(num_retrain_steps-1)]
retrain_steps = stack(n -> ceil(Int64, n), retrain_steps)
prior_shape = 0.01
prior_scale = 0.01
min_data_hard = 2
min_data_ratio = 1.0

penalty_vals = LinRange(0.01, 2.0, 10)
inflation_vals = LinRange(1.0, 15.0, 10)
regrets = zeros(length(inflation_vals), length(penalty_vals))
sds = zeros(length(inflation_vals), length(penalty_vals))
count = 1
total = length(penalty_vals) * length(inflation_vals) * num_sims
for (i, pen) in enumerate(penalty_vals), (j, infl) in enumerate(inflation_vals)
    rg = zeros(num_sims)
    for k in 1:num_sims
        print("\n\nSimulation $count/$total\n\n")
        env = sample_non_linear_env()
        pol = PartitionedPolyTS(
            get_num_dims(env),
            get_num_acts(env),
            initial_steps,
            retrain_steps;
            inflation=infl, # Iterating over this.
            max_models=200,
            max_degree=5,
            num_bfuns=15,
            min_data_hard=min_data_hard,
            min_data_ratio=min_data_ratio,
            penalty=pen, # Iterating over this.
            prior_shape=prior_shape,
            prior_scale=prior_scale
        )
        rg[k] = sum(run!(env, pol, num_steps; verbose=true))
        global count += 1
    end
    regrets[j, i], sds[j, i] = mean(rg), std(rg)
end
rg_plt = contourf(penalty_vals, inflation_vals, regrets, xlabel="\\gamma", ylabel="\\zeta")
sd_plt = contourf(penalty_vals, inflation_vals, sds, xlabel="\\gamma", ylabel="\\zeta")
try mkdir("figures") catch end
savefig(rg_plt, "figures/rg.png")
savefig(sd_plt, "figures/sd.png")

# Fix penalty and inflation values, then compare remaining hyperparameters
max_degs = collect(1:8)
num_bfs = collect(10:5:50)
regrets = zeros(length(num_bfs), length(max_degs))
sds = zeros(length(num_bfs), length(max_degs))
count = 1
total = length(max_degs) * length(num_bfs) * num_sims
for (i, deg) in enumerate(max_degs), (j, bf) in enumerate(num_bfs)
    rg = zeros(num_sims)
    for k in 1:num_sims
        print("\n\nSimulation $count/$total\n\n")
        env = sample_non_linear_env()
        pol = PartitionedPolyTS(
            get_num_dims(env),
            get_num_acts(env),
            initial_steps,
            retrain_steps;
            inflation=10.0,
            max_models=200,
            max_degree=deg,
            num_bfuns=bf,
            min_data_hard=min_data_hard,
            min_data_ratio=min_data_ratio,
            penalty=0.01,
            prior_shape=prior_shape,
            prior_scale=prior_scale
        )
        rg[k] = sum(run!(env, pol, num_steps; verbose=true))
        global count += 1
    end
    regrets[j, i], sds[j, i] = mean(rg), std(rg)
end
rg_plt = contourf(max_degs, num_bfs, regrets, xlabel="L", ylabel="H")
sd_plt = contourf(max_degs, num_bfs, sds, xlabel="L", ylabel="H")
try mkdir("figures") catch end
savefig(rg_plt, "figures/rg_H_L.png")
savefig(sd_plt, "figures/sd_H_L.png")