# Example: Simulated annealing - parameter optimization for denoising function
include(joinpath(@__DIR__, "..", "src", "annealing.jl"))
include(joinpath(@__DIR__, "..", "lib", "utils.jl"))
using .annealing
using .utils
using Random
using Images
using FileIO
using Plots

# Ensure results directory exists
results_dir = joinpath(@__DIR__, "..", "results")
if !isdir(results_dir)
    mkpath(results_dir)
end

# Generate true and noisy images
true_img = falses(100, 100)
true_img[30:50, 30:50] .= true
noisy_img = copy(true_img)
rand_mask = rand(100, 100) .< 0.1
noisy_img .= xor.(noisy_img, rand_mask)

# Initial parameter state
initial_params = (
    alpha=1.0,
    beta=2.0,
    T_init=2.0,
    cooling=0.99,
    max_iter=100_000
)

# Calculated parameter state 1
calculated_params_1 = (
    alpha=0.9269626427473135,
    beta=2.182786207341008,
    T_init=1.942096712167251,
    cooling=0.999868991320628,
    max_iter=97667
)

# Energy function for parameter optimization
param_energy_fn = p -> param_energy(p, true_img, noisy_img)

# Run parameter search via simulated annealing with logging
start_time = time()
best, log = simulated_annealing_log(
    initial_params,
    param_energy_fn,
    neighbor_params;
    T_init=2.0,
    cooling=0.99,
    max_iter=200,
    log_interval=1
)
total_time = time() - start_time
println("Parameter optimization completed in ", total_time, " seconds")

# Compute fit percentage
img_pixels = size(true_img,1)*size(true_img,2)
best_fit = 100 * (1 - param_energy_fn(best)/img_pixels)
println("Best fit: ", best_fit, "%")
println("Optimized parameters: ", best)

# Extract iterations and fits
iters = [it for (it, _) in log]
fits = [100 * (1 - e/img_pixels) for (_, e) in log]

# Plot parameter optimization progress
plot(iters, fits,
    xlabel="Iteration", ylabel="Fit (%)",
    title="Parameter optimization progress",
    legend=false
)
pngpath = joinpath(results_dir, "parameter_search.png")
savefig(pngpath)
println("Saved plot to ", pngpath)
