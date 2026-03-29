# Example: compare custom vs Optim.jl Simulated Annealing over 1000 iterations
include(joinpath(@__DIR__, "..", "src", "annealing.jl"))
include(joinpath(@__DIR__, "..", "lib", "utils.jl"))
using .annealing
using .utils
using Random
using Optim
using Plots

# Ensure results directory exists
results_dir = joinpath(@__DIR__, "..", "results")
if !isdir(results_dir)
    mkpath(results_dir)
end

# Generate 5 true/noisy image pairs
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

# Objective: multi-image energy
param_energy_fn = p -> param_energy_multi(p, true_imgs, noisy_imgs)

# Common constants
img_pixels = size(true_imgs[1],1) * size(true_imgs[1],2)
max_mismatch = 2 * n * img_pixels


# 1. Custom implementation
start_custom = time()
best_custom, log_custom = simulated_annealing_log(
    initial_params,
    param_energy_fn,
    neighbor_params;
    T_init = 2.0,
    cooling = 0.99,
    max_iter = 200,
    log_interval = 1
)
time_custom = time() - start_custom

# Extract fit % over iterations
iters_custom = [it for (it, _) in log_custom]
energies_custom = [e for (_, e) in log_custom]
fits_custom = [100 * (1 - e/max_mismatch) for e in energies_custom]

# Plot custom
plot(iters_custom, fits_custom,
    xlabel="Iteration", ylabel="Fit (%)",
    title="Custom Simulated Annealing", legend=false)
custom_path = joinpath(results_dir, "annealing_custom.png")
savefig(custom_path)
println("Saved custom plot to ", custom_path)


# 2. Optim.jl implementation

# Flatten params and define bounds
p0 = [initial_params.alpha, initial_params.beta, initial_params.T_init, initial_params.cooling, initial_params.max_iter]
T = 2.0
cooling = 0.99
max_iter = 200
lower = [0.0, 0.0, 1e-3, 0.75, 1.0]
upper = [10.0, 10.0, 10.0, 1.0, 1e6]

# Wrapped objective
f = x -> param_energy_fn((alpha=x[1], beta=x[2], T_init=x[3], cooling=x[4], max_iter=round(Int, x[5])))

# Cooling schedule
schedule = iter -> T * cooling^(iter-1)

# Configure Optim.jl SA
method = SimulatedAnnealing(
    neighbor    = neighbor_params!,
    temperature = schedule,
    keep_best   = true
)

# Run and trace
start_lib = time()
res_lib = optimize(f, lower, upper, p0, method, Optim.Options(iterations=max_iter, store_trace=true))
time_lib = time() - start_lib
trace_lib = Optim.trace(res_lib)
log_lib = [(te.iteration, te.value) for te in trace_lib]

# Extract fit
iters_lib = [it for (it, _) in log_lib]
energies_lib = [e for (_, e) in log_lib]
fits_lib = [100 * (1 - e/max_mismatch) for e in energies_lib]

# Final params
best_vec_lib = Optim.minimizer(res_lib)
best_lib = (alpha=best_vec_lib[1], beta=best_vec_lib[2], T_init=best_vec_lib[3], cooling=best_vec_lib[4], max_iter=round(Int, best_vec_lib[5]))

# Plot library
plot(iters_lib, fits_lib,
    xlabel="Iteration", ylabel="Fit (%)",
    title="Optim.jl Simulated Annealing", legend=false)
library_path = joinpath(results_dir, "annealing_library.png")
savefig(library_path)
println("Saved library plot to ", library_path)


# 3. Combined plot and save
plot(iters_custom, fits_custom, label="Custom")
plot!(iters_lib, fits_lib, label="Optim.jl")
title!("Comparison of Simulated Annealing Approaches")
xlabel!("Iteration"); ylabel!("Fit (%)")
combined_path = joinpath(results_dir, "annealing_combined.png")
savefig(combined_path)
println("Saved combined plot to ", combined_path)


# 4. Print timing and results
println("Custom SA: total time ", time_custom, "s, avg ", round(time_custom/1000, digits=4), "s/iter, best params: ", best_custom)
println("Optim.jl SA: total time ", time_lib, "s, avg ", round(time_lib/1000, digits=4), "s/iter, best params: ", best_lib)
println("Final fit custom: ", fits_custom[end], "%   library: ", fits_lib[end], "%")
