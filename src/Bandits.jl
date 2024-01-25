module Bandits

import LinearAlgebra as Alg
import Base: ==
import Distributions as Dst
import Flux as Flx
using Random: randperm
using ArgCheck: @argcheck
using Suppressor: @suppress
using SpecialFunctions: loggamma
using GLMNet: glmnet

export gaussian_data
include("utils.jl")

export Environment, get_num_dims, get_num_acts, get_context, act!, Policy, choice,
       update!, run!
include("api.jl")

export GaussianEnv, WheelEnv, LatentGaussianEnv, LabelledDataEnv
include("envs.jl")

export LinearModel, fit!, predict, get_coefs, trunc_tensor_product_basis, HyperRectangle,
       PolyModel
include("linear_model.jl")
include("poly_model.jl")
include("partitioned_poly_model.jl")

export RandomPolicy, PartitionedPolyTS, NeuralLinear, NeighborUCB
include("random_policy.jl")
include("neighbor_ucb.jl")
include("partitioned_thompson_sampling.jl")
include("neural_linear.jl")

end # module Bandits