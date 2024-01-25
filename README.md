# PPTS-Experitments

Source code used to generate the numerical results in "Partitioned Polynomial Thompson Sampling for Contextual Multi-Armed Bandits".
 
### Instructions

* Install [Julia 1.9](https://julialang.org/downloads/oldreleases/) and add to PATH.
* Clone this repository and change into `PPTS-Experiments`.
* To genrerate a set of regret trajectories, run...
```bash
# Choose environment from [latent_linear, non_lienar, near_linear, wheel07, wheel09, shuttle]
julia run_comparisons.jl latent_linear 
```
* To summarise results, run...
```bash
# Results written to Results/summary.csv
julia summarise_comparisons.jl
```