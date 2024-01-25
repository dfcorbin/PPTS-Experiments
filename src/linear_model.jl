mutable struct LinearModel
    prior_shape::Float64
    prior_scale::Float64
    prior_prec::Alg.Symmetric{Float64,Matrix{Float64}}
    prior_cov::Alg.Symmetric{Float64,Matrix{Float64}}

    shape::Float64
    scale::Float64
    prec::Alg.Symmetric{Float64,Matrix{Float64}}
    cov::Alg.Symmetric{Float64,Matrix{Float64}}
    coefs::Vector{Float64}
end

function LinearModel(penalty::Vector{Float64}; prior_shape::Float64=0.01,
                     prior_scale::Float64=0.01)
    @argcheck all(penalty .> 0)
    @argcheck prior_shape > 0
    @argcheck prior_scale > 0
    prior_prec = Alg.Symmetric(Alg.diagm(penalty))
    prior_cov = Alg.Symmetric(Alg.diagm(1 ./ penalty))
    prec = deepcopy(prior_prec)
    cov = deepcopy(prior_cov)
    coefs = zeros(Float64, length(penalty))
    return LinearModel(prior_shape, prior_scale, prior_prec, prior_cov, prior_shape,
                       prior_scale, prec, cov, coefs)
end

function LinearModel(num_dims::Int64; penalty::Float64=0.01, prior_shape::Float64=0.01,
                     prior_scale::Float64=0.01)
    @argcheck num_dims > 0
    return LinearModel(fill(penalty, num_dims); prior_shape=prior_shape,
                       prior_scale=prior_scale)
end

get_prior_shape(model::LinearModel) = model.prior_shape
get_prior_scale(model::LinearModel) = model.prior_scale
get_prior_prec(model::LinearModel) = deepcopy(model.prior_prec)
get_prior_cov(model::LinearModel) = deepcopy(model.prior_cov)
get_shape(model::LinearModel) = model.shape
get_scale(model::LinearModel) = model.scale
get_prec(model::LinearModel) = deepcopy(model.prec)
get_cov(model::LinearModel) = deepcopy(model.cov)
get_coefs(model::LinearModel) = model.coefs
get_num_dims(model::LinearModel) = length(model.coefs)
predict(model::LinearModel, inputs::AbstractArray{Float64}) = inputs' * model.coefs

function fit!(model::LinearModel, inputs::AbstractMatrix{Float64},
              targets::AbstractVector{Float64})
    @argcheck get_num_dims(model) == size(inputs, 1)
    @argcheck size(inputs, 2) == length(targets)
    (length(targets) == 0) && return nothing

    new_prec = Alg.Symmetric(inputs * inputs' + model.prec)
    new_cov = Alg.Symmetric(inv(new_prec))
    new_coefs = new_cov * (inputs * targets + model.prec * model.coefs)
    new_shape = model.shape + 0.5 * size(inputs, 2)
    new_scale = model.scale + 0.5 * targets' * targets
    new_scale += 0.5 * model.coefs' * model.prec * model.coefs
    new_scale -= 0.5 * new_coefs' * new_prec * new_coefs

    model.prec = new_prec
    model.cov = new_cov
    model.coefs = new_coefs
    model.shape = new_shape
    model.scale = new_scale
    return nothing
end

function fit!(model::LinearModel, input::AbstractVector{Float64}, target::Float64)
    fit!(model, reshape(input, (:, 1)), [target])
    return nothing
end

# function fit!(model::LinearModel, input::AbstractVector{Float64}, target::Float64)
#     # For sequential updates we can use the Sherman-Morrison inverse.
#     @argcheck get_num_dims(model) == length(input)
#     design = input * input'
#     new_prec = Alg.Symmetric(design + model.prec)
#     new_cov = model.cov -
#               model.cov * design * model.cov / (1 + input' * model.cov * input)
#     new_cov = Alg.Symmetric(new_cov)
#     new_coefs = new_cov * (target * input + model.prec * model.coefs)
#     new_shape = model.shape + 0.5
#     new_scale = model.scale + 0.5 * target^2
#     new_scale += 0.5 * model.coefs' * model.prec * model.coefs
#     new_scale -= 0.5 * new_coefs' * new_prec * new_coefs

#     model.prec = new_prec
#     model.cov = new_cov
#     model.coefs = new_coefs
#     model.shape = new_shape
#     model.scale = new_scale
#     return nothing
# end
