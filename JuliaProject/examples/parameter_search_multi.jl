# Example: Parameter optimization for multiple images
include(joinpath(@__DIR__, "..", "src", "annealing.jl"))
include(joinpath(@__DIR__, "..", "lib", "utils.jl"))
using .annealing, .utils, Random, Plots

# Ensure results directory exists
results_dir = joinpath(@__DIR__, "..", "results")
if !isdir(results_dir)
    mkpath(results_dir)
end

# Generate true/noisy image pairs
n = 5
true_imgs = Vector{BitMatrix}(undef, n)
noisy_imgs = Vector{BitMatrix}(undef, n)
for i in 1:n
    img = falses(100, 100)
    offset = 10*i
    img[30+offset:50+offset, 30+offset:50+offset] .= true
    true_imgs[i] = img
    noisy = copy(img)
    noisy .= xor.(noisy, rand(100, 100) .< 0.1)
    noisy_imgs[i] = noisy
end

# Initial parameters
initial_params = (
    alpha=1.0,
    beta=2.0,
    T_init=2.0,
    cooling=0.99,
    max_iter=100_000
)

# Calculated parameter state 1
best_params = (
    alpha = 1.0112623359095188,
    beta = 1.9757621741043059,
    T_init = 1.8676260258691217,
    cooling = 0.9999826101573366,
    max_iter = 100567)

# Calculated parameter state 2
best_params_2 = (
    alpha = 0.7789417503407984,
    beta = 2.232950776785252,
    T_init = 1.8805286659774354,
    cooling = 0.9999713548652925,
    max_iter = 101858
    )

# Energy function for parameter optimization (multi-image)
param_energy_fn = p -> param_energy_multi(p, true_imgs, noisy_imgs)

# Run annealing search and log progress
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
img_pixels = size(true_imgs[1],1)*size(true_imgs[1],2)
max_mismatch = 2 * n * img_pixels
best_fit = 100 * (1 - param_energy_fn(best)/max_mismatch)
println("Best fit: ", best_fit, "%")
println("Optimized parameters: ", best)

# Create a plot
iters = [it for (it,_) in log]
energies = [e for (_,e) in log]
fits = [100*(1 - e/max_mismatch) for e in energies]
plot(iters, fits,
    xlabel="Iteration", ylabel="Fit (%)",
    title="Parameter optimization progress",
    legend=false)
pngpath = joinpath(results_dir, "parameter_search_multi.png")
savefig(pngpath)
println("Saved plot to ", pngpath)
