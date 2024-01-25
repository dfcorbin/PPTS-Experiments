using Pkg: Pkg
Pkg.activate(".")
Pkg.resolve()
Pkg.instantiate()

using DataFrames: DataFrame, sort!
using Distributions: mean, median, std
using CSV: CSV

print("\nSumnmarsing results...\n")

file_names = readdir("Results/")
env_names, pol_names = String[], String[]
for fn in file_names
    fn == "summary.csv" && continue
    env_nm, pol_nm = split(fn, "-")
    pol_nm = replace(pol_nm, ".csv" => "")
    push!(env_names, env_nm)
    push!(pol_names, pol_nm)
end
env_names, pol_names = unique(env_names), unique(pol_names)
!("random" in pol_names) && error("Random policy is not found.")

results = DataFrame(
    :Environment => [],
    :Policy => [],
    :N => [],
    :Mean => [],
    :Median => [],
    :Std => []
)

for env_nm in env_names
    rnd_regret_matrix = Matrix(
        CSV.read("Results/" * env_nm * "-" * "random.csv", DataFrame; header = false)
    )
    rnd_regret_sums = sum(rnd_regret_matrix; dims = 2)

    for pol_nm in pol_names
        pol_nm == "random" && continue
        regret_matrix = Matrix(
            CSV.read(
                "Results/" * env_nm * "-" * pol_nm * ".csv",
                DataFrame;
                header = false
            )
        )
        regret_sums = sum(regret_matrix; dims = 2)
        num_sims = min(length(rnd_regret_sums), length(regret_sums), 100)
        regret_sums = 100 * regret_sums[1:num_sims] ./ rnd_regret_sums[1:num_sims]
        new_row = Dict(
            :Environment => env_nm,
            :Policy => pol_nm,
            :N => num_sims,
            :Mean => round(mean(regret_sums), sigdigits = 4),
            :Median => round(median(regret_sums), sigdigits = 4),
            :Std => round(std(regret_sums), sigdigits = 4)
        )
        push!(results, new_row)
    end
end

sort!(results, [:Environment, :Mean])
CSV.write("Results/summary.csv", results)

print("\nWrote results to Results/summary.csv\n")
