mutable struct NeighborUCB <: Policy
    act_contexts::Vector{Matrix{Float64}}
    scaled_act_contexts::Vector{Matrix{Float64}}
    act_rewards::Vector{Vector{Float64}}
    theta::Float64
    varphi::Float64
    step::Int64
    num_initial_steps::Int64

    # Used to standardize the data.
    ctex_mean::Vector{Float64}
    ctex_std::Vector{Float64}
    retrain_steps::Vector{Int64}

    function NeighborUCB(
        num_dims::Int64,
        num_acts::Int64,
        num_initial_steps::Int64,
        retrain_steps::Vector{Int64};
        theta::Float64=1.0,
        varphi::Float64=1.0,
    )
        @argcheck num_dims > 0
        @argcheck num_acts > 0
        @argcheck num_initial_steps >= num_acts
        @argcheck theta > 0
        @argcheck varphi > 0
        act_contexts = [Matrix{Float64}(undef, num_dims, 0) for _ in 1:num_acts]
        scaled_act_contexts = [Matrix{Float64}(undef, num_dims, 0) for _ in 1:num_acts]
        act_rewards = [Vector{Float64}(undef, 0) for _ in 1:num_acts]
        ctex_mean = zeros(Float64, num_dims)
        ctex_std = zeros(Float64, num_dims)
        return new(
            act_contexts,
            scaled_act_contexts,
            act_rewards,
            theta,
            varphi,
            0,
            num_initial_steps,
            ctex_mean,
            ctex_std,
            retrain_steps
        )
    end
end

function choice(policy::NeighborUCB, context::Vector{Float64})
    num_acts = length(policy.act_contexts)
    (policy.step <= policy.num_initial_steps) && return 1 + policy.step % num_acts

    upper_bounds = Vector{Float64}(undef, num_acts)
    scaled_ctex = (context - policy.ctex_mean) ./ policy.ctex_std
    for action in 1:num_acts
        distances = mapslices(policy.scaled_act_contexts[action]; dims=1) do neighbor
            return sqrt(sum((neighbor - scaled_ctex) .^ 2))
        end
        distances = vec(distances)
        perm = sortperm(distances)

        uncerts = Vector{Float64}(undef, length(policy.act_rewards[action]))
        for k in eachindex(distances)
            uncerts[k] = sqrt(policy.theta * log(policy.step) / k)
            uncerts[k] += policy.varphi * log(policy.step) * distances[perm[k]]
        end
        min_uncert, min_k = findmin(uncerts)

        upper_bounds[action] = sum(policy.act_rewards[action][perm[1:min_k]]) / min_k
        upper_bounds[action] += min_uncert
    end
    return argmax(upper_bounds)
end

function update!(policy::NeighborUCB, context::Vector{Float64}, action::Int64, reward::Float64)
    policy.step += 1
    policy.act_contexts[action] = hcat(policy.act_contexts[action], context)
    push!(policy.act_rewards[action], reward)

    if policy.step in policy.retrain_steps
        # Update the standardization parameters.
        full_ctex = hcat(policy.act_contexts...)
        policy.ctex_mean = vec(Dst.mean(full_ctex, dims=2))
        policy.ctex_std = vec(Dst.std(full_ctex, dims=2))
        any(policy.ctex_std .== 0.0) && error("Found feature with zero variance.")
        for a in eachindex(policy.scaled_act_contexts)
            policy.scaled_act_contexts[a] = (policy.act_contexts[a] .- policy.ctex_mean) ./ policy.ctex_std
        end
    else
        scaled_ctex = (context - policy.ctex_mean) ./ policy.ctex_std
        policy.scaled_act_contexts[action] = hcat(policy.scaled_act_contexts[action], scaled_ctex)
    end

    return nothing
end
