"""
    gaussian_data(mean_fn::Function, num_dims::Int64, num_obs::Int64)
        
Generate uniform inputs and Gaussian targets.

# Arguments

- `noise::Float64=1.0`: Standard deviation of the Gaussian noise.
- `lower::Float64=-1.0`: Lower bound of the input space.
- `upper::Float64=1.0`: Upper bound of the input space.
"""
function gaussian_data(mean_fn::Function, num_dims::Int64, num_obs::Int64;
                       noise::Float64=1.0, lower::Float64=-1.0, upper::Float64=1.0)
    inputs = lower .+ (upper - lower) * rand(num_dims, num_obs)
    targets = stack(mean_fn, eachcol(inputs)) + noise * randn(num_obs)
    return inputs, targets 
end