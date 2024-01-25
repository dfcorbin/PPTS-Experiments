mutable struct SplitNode
    key::Int64
    space::HyperRectangle

    dim::Int64
    loc::Float64
    left::SplitNode
    right::SplitNode

    SplitNode(key::Int64, space::HyperRectangle) = new(key, space)
end

mutable struct Partition
    tree::SplitNode
    leaves::Vector{SplitNode}
end

get_space(prt::Partition, key::Int64) = deepcopy(prt.leaves[key].space)

function Partition(space::HyperRectangle)
    tree = SplitNode(1, deepcopy(space))
    return Partition(tree, [tree])
end

function split!(prt::Partition, key::Int64, dim::Int64)
    leaf = prt.leaves[key]
    left_space = deepcopy(leaf.space)
    right_space = deepcopy(leaf.space)
    loc = sum(get_dim_lims(leaf.space, dim)) * 0.5
    set_upper_dim_lim!(left_space, dim, loc)
    set_lower_dim_lim!(right_space, dim, loc)

    leaf.key = 0
    leaf.dim = dim
    leaf.loc = loc
    leaf.left = SplitNode(key, left_space)
    leaf.right = SplitNode(length(prt.leaves) + 1, right_space)

    prt.leaves[key] = leaf.left
    push!(prt.leaves, leaf.right)
    return nothing
end

function find_key(node::SplitNode, input::AbstractVector{Float64})
    (node.key != 0) && return node.key
    if input[node.dim] < node.loc
        find_key(node.left, input)
    else
        find_key(node.right, input)
    end
end

find_key(prt::Partition, input::AbstractVector{Float64}) = find_key(prt.tree, input)

function find_key(prt::Partition, inputs::AbstractMatrix{Float64})
    return vec(mapslices(inp -> find_key(prt.tree, inp), inputs; dims = 1))
end

function lasso_poly_model(
    inputs::Matrix{Float64},
    targets::Vector{Float64},
    degree::Int64,
    num_bfuns::Int64,
    space::HyperRectangle;
    penalty::Float64 = 0.01,
    intercept_penalty::Float64 = penalty,
    prior_shape::Float64 = 0.01,
    prior_scale::Float64 = 0.01
)
    @argcheck size(inputs, 2) >= 2
    @argcheck degree >= 0
    @argcheck num_bfuns > 0

    # Set up basis expansion (excluding intercept).
    num_dims = size(inputs, 1)
    basis = trunc_tensor_product_basis(num_dims, degree)
    if basis[1] != MVIndex(Int64[], Int64[])
        mssg = "Expected trunc_tensor_product_basis to place intercept as the first element of the "
        mssg *= "basis."
        error(mssg)
    end

    if num_bfuns == 1
        basis = basis[1:1] # Always include intercept.
    elseif length(basis) > num_bfuns
        basis = basis[2:end] # Temporarily remove the intercept.
        features = legendre_expansion(inputs, basis, space)
        path = @suppress glmnet(
            features',
            targets;
            pmax = num_bfuns - 1, # Subtract 1 as we include the intercept by default.
            standardize = true,
            intercept = true
        )
        basis = basis[path.betas[:, end].!=0] # Extract sparse basis.
        pushfirst!(basis, MVIndex(Int64[], Int64[])) # Re-introduce intercept.
    end

    # For our application, it is useful to apply a different penalty to the intercept.
    penalty_vec = fill(penalty, length(basis))
    penalty_vec[1] = intercept_penalty
    model = PolyModel(
        basis,
        space,
        penalty_vec;
        prior_shape = prior_shape,
        prior_scale = prior_scale
    )
    fit!(model, inputs, targets)
    return model
end

function combined_evidence(sub_models::Vector{PolyModel})
    prior_shape = get_prior_shape(sub_models[1])
    prior_scale = get_prior_scale(sub_models[1])
    shape, scale, evidence = prior_scale, prior_shape, 0.0

    for model in sub_models
        if get_prior_shape(model) != prior_shape || get_prior_scale(model) != prior_scale
            msg = "Prior shape/scale must be constant across all sub-models."
            error(msg)
        end

        shape += get_shape(model) - prior_shape
        scale += get_scale(model) - prior_scale
        evidence += 0.5 * Alg.logdet(get_prior_prec(model))
        evidence -= 0.5 * Alg.logdet(get_prec(model))
    end

    num_obs = 2 * (shape - prior_shape)
    evidence -= 0.5 * num_obs * log(2 * π)
    evidence += prior_shape * log(prior_scale) - shape * log(scale)
    evidence += loggamma(shape) - loggamma(prior_shape)
    return evidence
end

function choose_subregion_split!(
    model_cache::Vector{Array{PolyModel,3}},
    key::Int64,
    models::Vector{PolyModel},
    min_obs::Vector{Float64},
    inputs::Matrix{Float64},
    targets::Vector{Float64},
    max_degree::Int64,
    num_bfuns::Int64,
    prior_shape::Float64,
    prior_scale::Float64,
    penalty::Float64,
    full_vol::Float64
)
    num_dims = size(inputs, 1)
    space = models[key].space
    tmp_models = deepcopy(models)
    push!(tmp_models, tmp_models[1]) # To add dummy entry to the end.
    split_info = Dict{String,Any}("evidence" => -Inf)

    for dim = 1:num_dims
        # Partition data
        loc = sum(get_dim_lims(space, dim)) * 0.5
        left_space = deepcopy(space)
        right_space = deepcopy(space)
        set_upper_dim_lim!(left_space, dim, loc)
        set_lower_dim_lim!(right_space, dim, loc)
        left_mask = inputs[dim, :] .< loc
        left_inputs, left_targets = inputs[:, left_mask], targets[left_mask]
        right_inputs, right_targets = inputs[:, .!left_mask], targets[.!left_mask]

        for ldeg = 0:max_degree, rdeg = 0:max_degree
            # Continue if insufficient data.
            (length(left_targets) < min_obs[ldeg+1]) && continue
            (length(right_targets) < min_obs[rdeg+1]) && continue

            # Add models to the cache if required.
            in_cache = isassigned(model_cache[key], 1, dim, ldeg + 1)
            in_cache &= isassigned(model_cache[key], 2, dim, rdeg + 1)
            if !in_cache
                model_cache[key][1, dim, ldeg+1] = lasso_poly_model(
                    left_inputs,
                    left_targets,
                    ldeg,
                    num_bfuns,
                    left_space;
                    penalty = penalty * full_vol / volume(left_space),
                    intercept_penalty = penalty,
                    prior_shape = prior_shape,
                    prior_scale = prior_scale
                )
                model_cache[key][2, dim, rdeg+1] = lasso_poly_model(
                    right_inputs,
                    right_targets,
                    rdeg,
                    num_bfuns,
                    right_space;
                    penalty = penalty * full_vol / volume(right_space),
                    intercept_penalty = penalty,
                    prior_scale = prior_shape,
                    prior_shape = prior_scale
                )
            end

            # Compute the evidence with the proposed models
            tmp_models[key] = model_cache[key][1, dim, ldeg+1]
            tmp_models[end] = model_cache[key][2, dim, rdeg+1]
            evidence = combined_evidence(tmp_models)
            if evidence > split_info["evidence"]
                split_info["evidence"] = evidence
                split_info["left_model"] = tmp_models[key]
                split_info["right_model"] = tmp_models[end]
                split_info["left_mask"] = left_mask
                split_info["dim"] = dim
            end
        end
    end
    return split_info["evidence"] == -Inf ? nothing : split_info
end

mutable struct PartitionedPolyModel
    sub_models::Vector{PolyModel}
    prt::Partition

    function PartitionedPolyModel(sub_models::Vector{PolyModel}, prt::Partition)
        prior_shape, prior_scale =
            get_prior_shape(sub_models[1]), get_prior_scale(sub_models[1])
        msg = "Number of sub-models does not match the number of sub-regions."
        (length(sub_models) != length(prt.leaves)) && error(msg)
        for (model, leaf) in zip(sub_models, prt.leaves)
            (model.space != leaf.space) && error("Model space does not match leaf space.")
            msg = "All sub-models must have the same prior shape."
            (get_prior_shape(model) != prior_shape) && error(msg)
            msg = "All sub-models must have the same prior scale."
            (get_prior_scale(model) != prior_scale) && error(msg)
        end
        return new(sub_models, prt)
    end
end

function choose_partitioned_basis(
    inputs::Matrix{Float64},
    targets::Vector{Float64},
    space::HyperRectangle;
    max_models::Int64 = 200,
    max_degree::Int64 = 5,
    num_bfuns::Int64 = 15,
    min_data_hard::Int64 = 2,
    min_data_ratio::Float64 = 1.0,
    prior_shape::Float64 = 0.01,
    prior_scale::Float64 = 0.01,
    penalty::Float64 = 0.01
)
    @argcheck size(inputs, 2) == length(targets)
    @argcheck length(targets) >= min_data_hard
    @argcheck get_num_dims(space) == size(inputs, 1)
    @argcheck max_models > 0
    @argcheck max_degree >= 0
    @argcheck num_bfuns >= 1
    @argcheck min_data_hard >= 2
    @argcheck min_data_ratio > 0

    # Initial setup
    num_dims, num_obs = size(inputs)
    min_obs = [
        max(
            min_data_hard,
            length(trunc_tensor_product_basis(num_dims, deg)) * min_data_ratio
        ) for deg = 0:max_degree
    ]
    models = Vector{PolyModel}(undef, 1)
    evidence = -Inf
    for deg = 0:max_degree
        num_obs < min_obs[deg+1] && continue
        m = lasso_poly_model(
            inputs,
            targets,
            deg,
            num_bfuns,
            space;
            prior_shape = prior_shape,
            prior_scale = prior_scale,
            penalty = penalty
        )
        tmp_evidence = combined_evidence([m])
        if tmp_evidence > evidence
            models[1] = m
            evidence = tmp_evidence
        end
    end

    model_cache = [Array{PolyModel,3}(undef, 2, num_dims, max_degree + 1)]
    keys = ones(Int64, num_obs)
    full_vol = volume(space)
    prt = Partition(space)

    while length(models) < max_models
        accepted = false
        for k in randperm(length(models))
            subregion_mask = keys .== k
            split_info = choose_subregion_split!(
                model_cache,
                k,
                models,
                min_obs,
                inputs[:, subregion_mask],
                targets[subregion_mask],
                max_degree,
                num_bfuns,
                prior_shape,
                prior_scale,
                penalty,
                full_vol
            )
            isnothing(split_info) && continue

            if split_info["evidence"] > evidence
                evidence = split_info["evidence"]
                models[k] = split_info["left_model"]
                push!(models, split_info["right_model"])
                split!(prt, k, split_info["dim"])
                subregion_keys = @view keys[subregion_mask]
                subregion_keys[.!split_info["left_mask"]] .= length(models)
                model_cache[k] = Array{PolyModel,3}(undef, 2, num_dims, max_degree + 1)
                push!(model_cache, Array{PolyModel,3}(undef, 2, num_dims, max_degree + 1))
                accepted = true
                break
            end
        end
        !accepted && break
    end
    return PartitionedPolyModel(models, prt)
end

function predict(model::PartitionedPolyModel, inputs::AbstractMatrix{Float64})
    keys = find_key(model.prt, inputs)
    targets = zeros(size(inputs, 2))
    for k in unique(keys)
        mask = keys .== k
        targets[mask] = predict(model.sub_models[k], inputs[:, mask])
    end
    return targets
end

function fit!(
    model::PartitionedPolyModel,
    inputs::Matrix{Float64},
    targets::Vector{Float64}
)
    locations = find_key(model.prt, inputs)
    for key in unique(locations)
        mask = locations .== key
        fit!(model.sub_models[key], inputs[:, mask], targets[mask])
    end
    return nothing
end

function fit!(model::PartitionedPolyModel, input::Vector{Float64}, target::Float64)
    key = find_key(model.prt, input)
    fit!(model.sub_models[key], input, target)
    return nothing
end

get_prior_shape(model::PartitionedPolyModel) = get_prior_shape(model.sub_models[1])
get_prior_scale(model::PartitionedPolyModel) = get_prior_scale(model.sub_models[1])
function get_prior_prec(model::PartitionedPolyModel, key::Int64)
    return get_prior_prec(model.sub_models[key])
end
function get_prior_cov(model::PartitionedPolyModel, key::Int64)
    return get_prior_cov(model.sub_models[key])
end
get_prec(model::PartitionedPolyModel, key::Int64) = get_prec(model.sub_models[key])
get_cov(model::PartitionedPolyModel, key::Int64) = get_cov(model.sub_models[key])
get_coefs(model::PartitionedPolyModel, key::Int64) = get_coefs(model.sub_models[key])
get_partition(model::PartitionedPolyModel) = deepcopy(model.prt)
get_basis(model::PartitionedPolyModel, key::Int64) = get_basis(model.sub_models[key])

function get_shape(model::PartitionedPolyModel)
    prior_shape = get_prior_shape(model)
    shape = prior_shape
    for m in model.sub_models
        shape += get_shape(m) - prior_shape
    end
    return shape
end

function get_scale(model::PartitionedPolyModel)
    prior_scale = get_prior_scale(model)
    scale = prior_scale
    for m in model.sub_models
        scale += get_scale(m) - prior_scale
    end
    return scale
end