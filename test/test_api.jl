mutable struct DummyEnv <: Environment
    ctex::Float64
end

Bandits.get_num_dims(env::DummyEnv) = 1
Bandits.get_num_acts(env::DummyEnv) = 2
Bandits.get_context(env::DummyEnv) = [env.ctex]

function Bandits.act!(env::DummyEnv, action::Int64)
    rewards = (env.ctex, -env.ctex)
    env.ctex += 1
    return rewards[action], maximum(rewards) - rewards[action]
end

mutable struct DummyPol <: Policy
    step::Int64
end

Bandits.choice(pol::DummyPol, ::Vector{Float64}) = mod(pol.step, 2) + 1

function Bandits.update!(pol::DummyPol, ::Vector{Float64}, ::Int64, ::Float64)
    pol.step += 1
    return nothing
end

@testset "CMAB trajectory" begin
    env = DummyEnv(-1.0)
    pol = DummyPol(1) # Chooses 2, then 1, then 2, ...
    regrets = run!(env, pol, 3)
    @test regrets == [0.0, 0.0, 2.0]
end