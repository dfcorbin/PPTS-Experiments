"""
Contextual multi-armed bandit environment.
"""
abstract type Environment end

"""
    get_num_dims(env::Environment)

Return the dimension of the context space.
"""
function get_num_dims end

"""
    get_num_acts(env::Environment)

Return the number of actions.
"""
function get_num_acts end

"""
    get_context(env::Environment)

Return the current environment context.
"""
function get_context end

"""
    act!(env::Environment, action::Int64)

Perform an action in the environment and update the context internally, then return an
Tuple{Float64, Float64} containing the regret and reward respectively.
"""
function act! end

"""
Contextual mulit-armed bandit policy.
"""
abstract type Policy end

"""
    choice(pol::Policy, ctex::Vector{Float64})

Return the policy`s choice of action
"""
function choice end

"""
    update!(pol::Policy, ctex::Vector{Float64}, action::Int64, reward::Float64)

Update the policy with a new context-action-reward triplet.
"""
function update! end

"""
    run!(env::Environment, pol::Policy, num_steps::Int64)

Run a contextual multi-armed bandit simulation and return the regret trajectory.
"""
function run!(env::Environment, pol::Policy, num_steps::Int64; verbose::Bool=false)
    regrets = Vector{Float64}(undef, num_steps)
    for step in 1:num_steps
        verbose && print("\rStep = $step/$num_steps")
        ctex::Vector{Float64} = get_context(env)
        action::Int64 = choice(pol, ctex)
        reward::Float64, regrets[step]::Float64 = act!(env, action)
        update!(pol, ctex, action, reward)
    end
    return regrets
end