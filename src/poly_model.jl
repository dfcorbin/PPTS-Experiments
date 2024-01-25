mutable struct HyperRectangle
    lower::Vector{Float64}
    upper::Vector{Float64}
end

function HyperRectangle(num_dims::Int64; lower::Float64=-1.0, upper::Float64=1.0)
    @argcheck num_dims > 0
    @argcheck lower <= upper
    return HyperRectangle(fill(lower, num_dims), fill(upper, num_dims))
end

get_num_dims(space::HyperRectangle) = length(space.lower)
get_dim_lims(space::HyperRectangle, dim::Int64) = (space.lower[dim], space.upper[dim])
volume(space::HyperRectangle) = prod(space.upper .- space.lower)

function ==(space1::HyperRectangle, space2::HyperRectangle)
    return (space1.lower == space2.lower) && (space1.upper == space2.upper)
end

function set_lower_dim_lim!(space::HyperRectangle, dim::Int64, loc::Float64)
    upper = get_dim_lims(space, dim)[2]
    if loc > upper
        throw(DomainError(loc, "New lower limit must be ≤ $upper."))
    end
    return space.lower[dim] = loc
end

function set_upper_dim_lim!(space::HyperRectangle, dim::Int64, loc::Float64)
    lower = get_dim_lims(space, dim)[1]
    if loc < lower
        throw(DomainError(loc, "New upper limit must be ≥ $lower"))
    end
    return space.upper[dim] = loc
end

function legendre_recursion(x::Float64, degree::Int64, p1::Float64, p0::Float64)
    (degree == 0) && return 1.0
    (degree == 1) && return x
    return (2 * degree - 1) * x * p1 / degree - (degree - 1) * p0 / degree
end

struct MVIndex
    dims::Vector{Int64}
    degs::Vector{Int64}
end

index_degree(idx::MVIndex) = length(idx.dims) == 0 ? 0 : maximum(idx.degs)
==(idx1::MVIndex, idx2::MVIndex) = (idx1.dims == idx2.dims) && (idx1.degs == idx2.degs)

function fill_basis!(num_dims::Int64, degree::Int64, current_dim::Int64,
                     index_dims::Vector{Int64}, index_degs::Vector{Int64},
                     basis::Vector{MVIndex})
    # First check if we have:
    # - Reached the maximum multivariate degree of the index.
    # - Finished assigning degrees to each dimension.
    current_deg = sum(index_degs)
    if current_deg == degree || current_dim == num_dims + 1
        push!(basis, MVIndex(index_dims, index_degs))
        return nothing
    end

    # Otherwise, append index vectors with every degree that does not exceed
    # the max_deg constraint and repeat for the next dimension (recursively).
    deg_remainder = degree - current_deg
    for uv_degree in 0:deg_remainder
        new_index_dims = deepcopy(index_dims)
        new_index_degs = deepcopy(index_degs)
        if uv_degree > 0
            push!(new_index_dims, current_dim)
            push!(new_index_degs, uv_degree)
        end
        fill_basis!(num_dims, degree, current_dim + 1, new_index_dims, new_index_degs,
                    basis)
    end
    return nothing
end

function trunc_tensor_product_basis(num_dims::Int64, degree::Int64)
    @argcheck num_dims > 0
    @argcheck degree >= 0
    basis = MVIndex[]
    fill_basis!(num_dims, degree, 1, Int64[], Int64[], basis)
    return basis
end

function legendre_expansion(inputs::AbstractMatrix{Float64}, basis::Vector{MVIndex},
                            space::HyperRectangle)
    num_dims, num_inputs = size(inputs)
    num_basis_fns = length(basis)
    if (num_inputs == 0) || (num_basis_fns == 0)
        return Matrix{Float64}(undef, num_basis_fns, num_inputs)
    end

    degree = maximum(index_degree.(basis))
    uv_features = Matrix{Float64}(undef, num_dims, degree + 1)
    uv_features[:, 1] .= 1.0
    mv_features = ones(Float64, num_basis_fns, num_inputs)

    for i in 1:num_inputs
        # First the univariate expansions are computed and stored for the current input (column).
        for dim in 1:num_dims
            if degree == 0
                continue
            end

            # Rescale the feature onto its interval.
            x = inputs[dim, i]
            lower_dim_lim, upper_dim_lim = get_dim_lims(space, dim)
            if !(lower_dim_lim <= x <= upper_dim_lim)
                mssg = "The feature inputs[$dim, $i] is not contained by the space limits "
                mssg *= "($lower_dim_lim, $upper_dim_lim)."
                throw(DomainError(x, mssg))
            end
            x1 = (2 * x - lower_dim_lim - upper_dim_lim) / (upper_dim_lim - lower_dim_lim)

            # Degree 1 univariate expansion is already defined.
            uv_features[dim, 2] = x1
            degree == 1 && continue

            # Compute the remaining univariate expansions using the (order 2) Legendre recursion.
            for deg in 2:degree
                uv_features[dim, deg + 1] = legendre_recursion(x1, deg,
                                                               uv_features[dim, deg],
                                                               uv_features[dim, deg - 1])
            end
        end

        # The final features are computed as multiplicative combinations of the univariate features.
        for b in 1:num_basis_fns, (dim, deg) in zip(basis[b].dims, basis[b].degs)
            mv_features[b, i] *= uv_features[dim, deg + 1]
        end
    end
    return mv_features
end

function legendre_expansion(input::AbstractVector{Float64}, basis::Vector{MVIndex},
                            space::HyperRectangle)
    inputs = reshape(input, (:, 1))
    return vec(legendre_expansion(inputs, basis, space))
end

mutable struct PolyModel
    basis::Vector{MVIndex}
    space::HyperRectangle
    linear::LinearModel
end

function PolyModel(basis::Vector{MVIndex}, space::HyperRectangle,
                   penalty::Vector{Float64}; prior_shape::Float64=0.01, prior_scale=0.01)
    linear = LinearModel(penalty; prior_shape=prior_shape, prior_scale=prior_scale)
    return PolyModel(basis, space, linear)
end

function PolyModel(basis::Vector{MVIndex}, space::HyperRectangle; penalty::Float64=0.01,
                   prior_shape::Float64=0.01, prior_scale::Float64=0.01)
    linear = LinearModel(fill(penalty, length(basis)); prior_shape=prior_shape,
                         prior_scale=prior_scale)
    return PolyModel(basis, space, linear)
end

get_prior_shape(model::PolyModel) = get_prior_shape(model.linear)
get_prior_scale(model::PolyModel) = get_prior_scale(model.linear)
get_prior_prec(model::PolyModel) = get_prior_prec(model.linear)
get_prior_cov(model::PolyModel) = get_prior_cov(model.linear)
get_shape(model::PolyModel) = get_shape(model.linear)
get_scale(model::PolyModel) = get_scale(model.linear)
get_prec(model::PolyModel) = get_prec(model.linear)
get_cov(model::PolyModel) = get_cov(model.linear)
get_coefs(model::PolyModel) = get_coefs(model.linear)
variance(model::PolyModel) = variance(model.linear)
get_basis(model::PolyModel) = deepcopy(model.basis)

function fit!(model::PolyModel, inputs, targets)
    features = legendre_expansion(inputs, model.basis, model.space)
    fit!(model.linear, features, targets)
    return nothing
end

function predict(model::PolyModel, inputs::AbstractArray{Float64})
    features = legendre_expansion(inputs, model.basis, model.space)
    return features' * model.linear.coefs
end