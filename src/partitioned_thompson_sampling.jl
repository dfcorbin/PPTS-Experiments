function truncate_context(space::HyperRectangle, context::AbstractVector{Float64})
    num_dims = length(context)
    new_context = deepcopy(context)
    for dim = 1:num_dims
        lower, upper = get_dim_lims(space, dim)
        new_context[dim] = max(lower, new_context[dim])
        new_context[dim] = min(upper, new_context[dim])
    end
    return new_context
end

mutable struct PartitionedPolyTS <: Policy
    inputs::Matrix{Float64}
    actions::Vector{Int64}
    rewards::Vector{Float64}

    initial_steps::Int64
    retrain_steps::Vector{Int64}
    inflation::Float64

    act_models::Vector{PartitionedPolyModel}
    space::HyperRectangle
    space_cache::HyperRectangle
    max_models::Int64
    max_degree::Int64
    num_bfuns::Int64
    min_data_hard::Int64
    min_data_ratio::Float64
    penalty::Float64
    prior_shape::Float64
    prior_scale::Float64
end

function PartitionedPolyTS(
    num_dims::Int64,
    num_acts::Int64,
    initial_steps::Int64,
    retrain_steps::Vector{Int64};
    inflation::Float64 = 1.0,
    max_models::Int64 = 200,
    max_degree::Int64 = 5,
    num_bfuns::Int64 = 15,
    min_data_hard::Int64 = 2,
    min_data_ratio::Float64 = 1.0,
    penalty::Float64 = 0.01,
    prior_shape::Float64 = 0.01,
    prior_scale::Float64 = 0.01
)
    space = HyperRectangle(num_dims; lower = 0.0, upper = 0.0)
    space_cache = deepcopy(space)
    inputs = Matrix{Float64}(undef, num_dims, 0)
    actions = Int64[]
    rewards = Float64[]
    act_models = Vector{PartitionedPolyModel}(undef, num_acts)
    return PartitionedPolyTS(
        inputs,
        actions,
        rewards,
        initial_steps,
        retrain_steps,
        inflation,
        act_models,
        space,
        space_cache,
        max_models,
        max_degree,
        num_bfuns,
        min_data_hard,
        min_data_ratio,
        penalty,
        prior_shape,
        prior_scale
    )
end

function choice(pol::PartitionedPolyTS, context::Vector{Float64})
    num_acts = length(pol.act_models)
    step = length(pol.actions)
    (step <= pol.initial_steps) && return step % num_acts + 1

    trunc_ctex = truncate_context(pol.space, context)
    thompson_samples = Vector{Float64}(undef, num_acts)
    for action = 1:num_acts
        prt = get_partition(pol.act_models[action])
        key = find_key(prt, context)
        basis = get_basis(pol.act_models[action], key)
        space = get_space(prt, key)
        features = legendre_expansion(trunc_ctex, basis, space)
        coefs = get_coefs(pol.act_models[action], key)

        shape, scale =
            get_shape(pol.act_models[action]), get_scale(pol.act_models[action])
        var_sample = rand(Dst.InverseGamma(shape, scale))
        cov_sample = pol.inflation * var_sample * get_cov(pol.act_models[action], key)
        coefs_sample = rand(Dst.MvNormal(coefs, cov_sample))
        thompson_samples[action] = features' * coefs_sample
    end
    return argmax(thompson_samples)
end

function update!(
    pol::PartitionedPolyTS,
    context::Vector{Float64},
    action::Int64,
    reward::Float64
)
    pol.inputs = hcat(pol.inputs, context)
    push!(pol.actions, action)
    push!(pol.rewards, reward)

    for dim in eachindex(context)
        lower, upper = get_dim_lims(pol.space_cache, dim)
        new_lower = min(context[dim], lower)
        new_upper = max(context[dim], upper)
        set_lower_dim_lim!(pol.space_cache, dim, new_lower)
        set_upper_dim_lim!(pol.space_cache, dim, new_upper)
    end

    step = length(pol.actions)
    step < pol.initial_steps && return nothing
    if (step in pol.retrain_steps) || (step == pol.initial_steps)
        for a = 1:length(pol.act_models)
            pol.space = deepcopy(pol.space_cache)
            act_inputs = pol.inputs[:, pol.actions.==a]
            act_rewards = pol.rewards[pol.actions.==a]
            pol.act_models[a] = choose_partitioned_basis(
                act_inputs,
                act_rewards,
                pol.space;
                max_models = pol.max_models,
                max_degree = pol.max_degree,
                num_bfuns = pol.num_bfuns,
                min_data_hard = pol.min_data_hard,
                min_data_ratio = pol.min_data_ratio,
                penalty = pol.penalty,
                prior_shape = pol.prior_shape,
                prior_scale = pol.prior_scale
            )
        end
    else
        trunc_ctex = truncate_context(pol.space, context)
        fit!(pol.act_models[action], trunc_ctex, reward)
    end
end