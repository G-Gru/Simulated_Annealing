# Example: Parameter search using Optim.jl's Simulated Annealing
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

# Parameters for Optim.jl's Simulated Annealing
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

# Print results
println("Optim.jl SA: Best params: $best_lib, Time: $time_lib seconds")

# Plot Optim.jl results
plot(iters_lib, fits_lib,
    xlabel="Iteration", ylabel="Fit (%)",
    title="Optim.jl Simulated Annealing", legend=false)
pngpath = joinpath(results_dir, "parameter_search_optim.png")
savefig(pngpath)
println("Saved plot to ", pngpath)
