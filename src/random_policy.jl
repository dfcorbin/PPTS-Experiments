struct RandomPolicy <: Policy
    num_acts::Int64

    function RandomPolicy(num_acts::Int64)
        @argcheck num_acts > 0
        return new(num_acts)
    end
end

choice(policy::RandomPolicy, ::Vector{Float64}) = rand(1:(policy.num_acts))
update!(::RandomPolicy, ::Vector{Float64}, ::Int64, ::Float64) = nothing