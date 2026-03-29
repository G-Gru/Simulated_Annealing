# Example: Simulated Annealing - Denoising a Binary Image
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

# Create a binary/noisy image
binary_img = falses(100, 100)
binary_img[30:50, 30:50] .= true
noisy_img = copy(binary_img)
rand_mask = rand(100, 100) .< 0.1
noisy_img .= xor.(noisy_img, rand_mask)

# Visualize original/noisy image
orig_path = joinpath(results_dir, "original_img.png")
save(orig_path, Gray.(binary_img))
println("Original image saved to ", orig_path)
noisy_path = joinpath(results_dir, "noisy_img.png")
save(noisy_path, Gray.(noisy_img))
println("Noisy image saved to ", noisy_path)

# Define a unary energy function capturing the noisy image
energy_fn = I_new -> energy(I_new, noisy_img; alpha = 0.7789417503407984, beta = 2.2329507767852521125)

# Run simulated annealing denoising with logging
start_time = time()
best_img, log = simulated_annealing_log(
    noisy_img,
    energy_fn,
    default_neighbor;
    T_init = 1.8805286659774354,
    cooling = 0.9999713548652925,
    max_iter = 101858,
    log_interval = 10
)
total_time = time() - start_time
println("Denoising completed in ", total_time, " seconds")

# Visualize result
denoised_path = joinpath(results_dir, "denoised_img.png")
save(denoised_path, Gray.(best_img))
println("Denoised image saved to ", denoised_path)

# Extract iterations and energies
iters = [it for (it, _) in log]
energies = [e for (_, e) in log]

# Plot energy over iterations
plot(iters, energies,
    xlabel = "Iteration", ylabel = "Energy",
    title = "Denoising Progress",
    legend = false
)
energy_png = joinpath(results_dir, "denoising_progress.png")
savefig(energy_png)
println("Saved energy plot to ", energy_png)
