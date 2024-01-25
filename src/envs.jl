mutable struct GaussianEnv{T<:Function} <: Environment
    mean_fn::T
    num_acts::Int64
    noise::Float64
    lower::Float64
    upper::Float64
    ctex::Vector{Float64}

    function GaussianEnv(mean_fn::T, num_dims::Int64, num_acts::Int64; noise::Float64=1.0,
                         lower::Float64=0.0, upper::Float64=1.0) where {T<:Function}
        context = lower .+ (upper - lower) * rand(num_dims)
        return new{T}(mean_fn, num_acts, noise, lower, upper, context)
    end
end

get_context(env::GaussianEnv) = deepcopy(env.ctex)
get_num_dims(env::GaussianEnv) = length(env.ctex)
get_num_acts(env::GaussianEnv) = env.num_acts

function act!(env::GaussianEnv, action::Int64)
    !(1 <= action <= get_num_acts(env)) && throw(DomainError(action, "Invalid action."))
    means = [env.mean_fn(env.ctex, a)::Float64 for a in 1:(env.num_acts)]
    regret = maximum(means) - means[action]
    reward = means[action] + env.noise * randn()
    env.ctex = env.lower .+ (env.upper - env.lower) * rand(get_num_dims(env))
    return reward, regret
end

mutable struct WheelEnv <: Environment
    means::Tuple{Float64,Float64,Float64}
    noise::Float64
    radius::Float64
    ctex::Vector{Float64}
end

function WheelEnv(means::Tuple{Float64,Float64,Float64}, noise::Float64, radius::Float64)
    if !(means[3] > means[1] > means[2])
        throw(ArgumentError("Means ordered incorrectly."))
    elseif !(0 <= radius <= 1)
        throw(DomainError(radius, "Radius must be in (0, 1)."))
    end
    context_radius = sqrt(rand())
    context_angle = 2 * π * rand()
    context = [context_radius * cos(context_angle), context_radius * sin(context_angle)]
    return WheelEnv(means, noise, radius, context)
end

get_num_dims(::WheelEnv) = 2
get_num_acts(::WheelEnv) = 5
get_context(env::WheelEnv) = deepcopy(env.ctex)

function act!(env::WheelEnv, action::Int64)
    if !(1 <= action <= 5)
        throw(DomainError(action, "Invalid action."))
    end
    reward_noise = env.noise * randn()
    is_centre = sqrt(sum(env.ctex .^ 2)) <= env.radius

    if action == 1
        # Action 1 always returns env.means[1]. This is optimal within the centre cirlce,
        # but suboptimal in the outer circle.
        reward = env.means[1] + reward_noise
        regret = is_centre ? 0.0 : (env.means[3] - env.means[1])
    elseif is_centre
        # All remaining actions (2-5) return env.means[2] (suboptimal) within the centre
        # circle.
        reward = env.means[2] + reward_noise
        regret = env.means[1] - env.means[2]
    else
        # In the outer circle, each quadrant has a single optimal action returning 
        # env.means[3], the rest return env.means[2].
        is_optimal = ((action == 2) && (env.ctex[1] >= 0) && (env.ctex[2] >= 0) || # TR
                      (action == 3) && (env.ctex[1] >= 0) && (env.ctex[2] < 0) || # BR
                      (action == 4) && (env.ctex[1] < 0) && (env.ctex[2] >= 0) || #  TL
                      (action == 5) && (env.ctex[1] < 0) && (env.ctex[2] < 0)) # BL
        reward = (is_optimal ? env.means[3] : env.means[2]) + reward_noise
        regret = is_optimal ? 0.0 : (env.means[3] - env.means[2])
    end

    ctex_radius = sqrt(rand())
    ctex_angle = 2 * π * rand()
    env.ctex = [ctex_radius * cos(ctex_angle), ctex_radius * sin(ctex_angle)]
    return reward, regret
end

mutable struct LatentGaussianEnv{T<:Function} <: Environment
    tform::T
    latent_env::GaussianEnv
    ctex::Vector{Float64}
end

function LatentGaussianEnv(tform::Function, mean_fn::Function, num_latent_dims::Int64,
                           num_acts::Int64; noise::Float64=1.0, lower::Float64=0.0,
                           upper::Float64=1.0)
    latent_env = GaussianEnv(mean_fn, num_latent_dims, num_acts; noise=noise, lower=lower,
                             upper=upper)
    ctex = tform(get_context(latent_env))::Vector{Float64}
    return LatentGaussianEnv(tform, latent_env, ctex)
end

get_num_dims(env::LatentGaussianEnv) = length(env.ctex)
get_num_acts(env::LatentGaussianEnv) = get_num_acts(env.latent_env)
get_context(env::LatentGaussianEnv) = deepcopy(env.ctex)

function act!(env::LatentGaussianEnv, action::Int64)
    obs = act!(env.latent_env, action)
    env.ctex = env.tform(env.latent_env.ctex)
    return obs
end

mutable struct LabelledDataEnv <: Environment
    ctexs::Matrix{Float64}
    labels::Vector{Int64}
    step::Int64
    num_acts::Int64
end

function LabelledDataEnv(ctexs::Matrix{Float64}, labels::Vector{Int64})
    if any(size(ctexs) .== 0)
        throw(ArgumentError("Contexts must be non-empty."))
    elseif size(ctexs, 2) != length(labels)
        throw(DimensionMismatch("Number of contexts must equal number of labels."))
    elseif any(labels .<= 0)
        throw(DimensionMismatch("Labels must be positive."))
    end
    num_ctex = size(ctexs, 2)
    perm = randperm(num_ctex)
    shuffled_ctexs = ctexs[:, perm]
    shuffled_labels = labels[perm]
    num_acts = maximum(labels)
    return LabelledDataEnv(shuffled_ctexs, shuffled_labels, 1, num_acts)
end

get_num_dims(env::LabelledDataEnv) = size(env.ctexs, 1)
get_num_acts(env::LabelledDataEnv) = env.num_acts
get_context(env::LabelledDataEnv) = deepcopy(env.ctexs[:, env.step])

function act!(env::LabelledDataEnv, action::Int64)
    if !(1 <= action <= get_num_acts(env))
        throw(DomainError(action, "Invalid action."))
    elseif !(0 <= env.step <= length(env.labels))
        throw(DomainError(env.step, "Exceeded number of contexts."))
    end
    reward = Float64(action == env.labels[env.step])
    regret = 1 - reward
    env.step += 1
    return reward, regret
end