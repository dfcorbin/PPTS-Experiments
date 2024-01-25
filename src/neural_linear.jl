function encoder_network(num_dims::Int64, widths::Vector{Int64}, num_acts::Int64)
    (length(widths) == 1) && return Flx.Chain(;
        features=Flx.Dense(num_dims => widths[1], Flx.relu),
        means=Flx.Dense(widths[end] => num_acts),
    )
    return Flx.Chain(;
        features=Flx.Chain(
            Flx.Dense(num_dims => widths[1], Flx.relu),
            [Flx.Dense(w_in => w_out, Flx.relu) for (w_in, w_out) in zip(widths, widths[2:end])]...,
        ),
        means=Flx.Dense(widths[end] => num_acts),
    )
end

function action_weights(num_acts::Int64, actions::Vector{Int64})
    weight_mat = zeros(Float32, num_acts, length(actions))
    for i in eachindex(actions)
        weight_mat[actions[i], i] = 1.0
    end
    return weight_mat
end

function loss(
    network::Flx.Chain,
    contexts::AbstractMatrix{<:Real},
    weights::AbstractMatrix{<:Real},
    rewards::AbstractVector{<:Real},
)
    weighted_error = (network(contexts) .- rewards') .* weights
    return sum(weighted_error .^ 2) / length(rewards)
end

function train_encoder!(
    network::Flx.Chain,
    contexts::AbstractMatrix{<:Real},
    weights::AbstractMatrix{<:Real},
    rewards::AbstractVector{<:Real};
    num_epochs::Int64=50,
    batch_size::Int64=32,
    opt_state::Union{Nothing,NamedTuple}=nothing,
    opt_rule::Flx.Optimise.AbstractOptimiser=Flx.Adam(),
    verbose::Bool=false,
)
    opt_state = isnothing(opt_state) ? Flx.setup(opt_rule, network) : opt_state
    data = Flx.DataLoader(
        (contexts=contexts, weights=weights, rewards=rewards);
        batchsize=min(length(rewards), batch_size),
        shuffle=true,
    )
    losses = zeros(num_epochs)
    for epoch in 1:num_epochs
        for batch in data
            batch_loss, grads = Flx.withgradient(network) do net
                return loss(net, batch.contexts, batch.weights, batch.rewards)
            end
            losses[epoch] += batch_loss * length(batch.rewards) / length(rewards)
            Flx.update!(opt_state, network, grads[1])
        end
        verbose && print("\nepoch = $epoch/$num_epochs, loss = $(losses[epoch])")
    end
    return losses
end

mutable struct NeuralLinear{T<:Flx.Chain} <: Policy
    # Policy stores its observations for retraining.
    contexts::Matrix{Float64}
    actions::Vector{Int64}
    rewards::Vector{Float64}
    act_features::Vector{Matrix{Float64}}
    act_rewards::Vector{Vector{Float64}}

    initial_steps::Int64 # Number of steps with randomly chosen actions.
    retrain_steps::Vector{Int64} # Steps where retraining occurs.
    inflation::Float64 # Inflation parameter for Thompson sampling.

    encoder::T # Neural network used to output features for linear models.
    linear_models::Vector{LinearModel} # One linear model for each action.

    # Model (re)training parameters.
    widths::Vector{Int64}
    num_epochs::Int64
    batch_size::Int64
    penalty::Float64
    prior_shape::Float64
    prior_scale::Float64
end

function NeuralLinear(
    num_dims::Int64,
    num_acts::Int64,
    initial_steps::Int64,
    retrain_steps::Vector{Int64};
    inflation::Float64=1.0,
    widths::Vector{Int64}=[64, 64],
    num_epochs::Int64=50,
    batch_size::Int64=32,
    penalty::Float64=0.01,
    prior_shape::Float64=0.01,
    prior_scale::Float64=0.01,
)
    @argcheck initial_steps >= num_acts
    @argcheck length(widths) > 0
    @argcheck all(widths .> 0)
    @argcheck num_epochs > 0
    @argcheck batch_size > 0

    contexts = Matrix{Float64}(undef, num_dims, 0)
    actions = Int64[]
    rewards = Float64[]
    act_features = [Matrix{Float64}(undef, widths[end], 0) for _ in 1:num_acts]
    act_rewards = [Float64[] for _ in 1:num_acts]

    encoder = encoder_network(num_dims, widths, num_acts)
    linear_models = [
        LinearModel(widths[end]; penalty=penalty, prior_shape=prior_shape, prior_scale=prior_scale)
        for _ in 1:num_acts
    ]
    return NeuralLinear(
        contexts,
        actions,
        rewards,
        act_features,
        act_rewards,
        initial_steps,
        retrain_steps,
        inflation,
        encoder,
        linear_models,
        widths,
        num_epochs,
        batch_size,
        penalty,
        prior_shape,
        prior_scale,
    )
end

function update!(policy::NeuralLinear, context::Vector{Float64}, action::Int64, reward::Float64)
    # Store the new observation
    policy.contexts = hcat(policy.contexts, context)
    push!(policy.actions, action)
    push!(policy.rewards, reward)
    policy.act_features[action] = hcat(
        policy.act_features[action], policy.encoder[:features](context)
    )
    push!(policy.act_rewards[action], reward)

    step = length(policy.actions)
    step < policy.initial_steps && return nothing

    if step in policy.retrain_steps || step == policy.initial_steps
        # Retrain the encoder network from scratch
        num_dims = length(context)
        num_acts = length(policy.linear_models)
        new_encoder = encoder_network(num_dims, policy.widths, num_acts)
        train_encoder!(
            new_encoder,
            policy.contexts,
            action_weights(num_acts, policy.actions),
            policy.rewards;
            num_epochs=policy.num_epochs,
            batch_size=policy.batch_size,
            opt_rule=Flx.Adam(),
        )
        policy.encoder = new_encoder

        features = policy.encoder[:features](policy.contexts)
        features = Float64.(features)
        for a in 1:num_acts
            mask = (policy.actions .== a)
            policy.act_features[a] = features[:, mask]
            policy.linear_models[a] = LinearModel(
                policy.widths[end];
                penalty=policy.penalty,
                prior_shape=policy.prior_shape,
                prior_scale=policy.prior_scale,
            )
            fit!(policy.linear_models[a], features[:, mask], policy.rewards[mask])
        end
    else
        policy.linear_models[action] = LinearModel(
            policy.widths[end];
            penalty=policy.penalty,
            prior_shape=policy.prior_shape,
            prior_scale=policy.prior_scale,
        )
        fit!(policy.linear_models[action], policy.act_features[action], policy.act_rewards[action])
    end
    return nothing
end

function choice(policy::NeuralLinear, context::Vector{Float64})
    num_acts = length(policy.linear_models)
    step = length(policy.actions)
    step < policy.initial_steps && return (step % num_acts) + 1

    features = policy.encoder[:features](context)
    features = Float64.(features)
    samples = Vector{Float64}(undef, num_acts)
    for action in 1:num_acts
        model = policy.linear_models[action]
        shape, scale = get_shape(model), get_scale(model)
        var_sample = rand(Dst.InverseGamma(shape, scale))
        cov_sample = var_sample * policy.inflation * model.cov
        coefs_sample = rand(Dst.MvNormal(model.coefs, cov_sample))
        samples[action] = coefs_sample' * features
    end
    return argmax(samples)
end