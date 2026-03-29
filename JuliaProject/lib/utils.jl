# Functions used in the project
module utils

include(joinpath(@__DIR__, "..", "src", "annealing.jl"))
using .annealing, Random, Base.Threads

export energy, param_energy, neighbor_params, param_energy_multi, default_neighbor, neighbor_params!

# Energy function for binary image denoising
function energy(I::AbstractMatrix{Bool}, I_noisy::AbstractMatrix{Bool}; alpha::Float64=1.0, beta::Float64=2.0)
    E = alpha * sum(I .!= I_noisy)
    E += beta * sum(I[:, 1:end-1] .!= I[:, 2:end])   # horizontal
    E += beta * sum(I[1:end-1, :] .!= I[2:end, :])   # vertical
    return E
end

# Default neighbor function (used for denoising)
function default_neighbor(I::AbstractMatrix{Bool})
    # flip a random pixel
    newI = copy(I)
    h, w = size(I)
    x, y = rand(1:h), rand(1:w)
    newI[x, y] = !newI[x, y]
    return newI
end

# Energy on parameter state: unmatched pixels count
function param_energy(p, true_img, noisy_img)
    # energy_fn for this p using noisy_img as reference
    energy_fn = I_new -> energy(I_new, noisy_img; alpha=p.alpha, beta=p.beta)
    # Perform denoising
    result = simulated_annealing(noisy_img, energy_fn, default_neighbor;
        T_init=p.T_init,
        cooling=p.cooling,
        max_iter=p.max_iter
    )
    # Number of pixels that don't match the true image
    return sum(result .!= true_img)
end

# Energy on parameter state for multiple true images, each with its own noisy image, each used twice
function param_energy_multi(p, true_imgs::Vector{<:AbstractMatrix{Bool}}, noisy_imgs::Vector{<:AbstractMatrix{Bool}})
    n = length(true_imgs)
    mismatches = zeros(Int, n)
    @threads for i in 1:n
        local_mismatch = 0
        for _ in 1:2
            energy_fn = I_new -> energy(I_new, noisy_imgs[i]; alpha=p.alpha, beta=p.beta)
            result = simulated_annealing(noisy_imgs[i], energy_fn, default_neighbor;
                T_init=p.T_init,
                cooling=p.cooling,
                max_iter=p.max_iter
            )
            local_mismatch += sum(result .!= true_imgs[i])
        end
        mismatches[i] = local_mismatch
    end
    return sum(mismatches)
end

# Neighbor function for parameter state
function neighbor_params(p)
    return (
        alpha = max(0.0, p.alpha + 0.2*(rand()-0.5)),
        beta = max(0.0, p.beta + 0.2*(rand()-0.5)),
        T_init = max(1e-3, p.T_init + 0.2*(rand()-0.5)),
        cooling = clamp(p.cooling + 0.02*(rand()-0.5), 0.75, 0.999999999999),
        max_iter = max(1, p.max_iter + rand(-1000:1000))
    )
end

# Neighbor function for parameter state for library annealing
function neighbor_params!(x_current, x_proposed)
    # Mutate parameters
    x_proposed[1] = max(0.0, x_current[1] + 0.2*(rand()-0.5)) # alpha
    x_proposed[2] = max(0.0, x_current[2] + 0.2*(rand()-0.5)) # beta
    x_proposed[3] = max(1e-3, x_current[3] + 0.2*(rand()-0.5)) # T_init
    x_proposed[4] = clamp(x_current[4] + 0.02*(rand()-0.5), 0.75, 0.9999999999) # cooling
    x_proposed[5] = clamp(x_current[5] + rand(-1000:1000), 1.0, 1e6) # max_iter
end
end
