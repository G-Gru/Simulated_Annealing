# Example tests, not made to be practical
include(joinpath(@__DIR__, "..", "src", "annealing.jl"))
include(joinpath(@__DIR__, "..", "lib", "utils.jl"))
using .utils
using .annealing
using Test

# Test that simulated_annealing returns a state of the same type and shape as the input
@testset "simulated_annealing type/shape" begin
    using ..utils
    img = falses(10, 10)
    img[3:7, 3:7] .= true
    noisy_img = copy(img)
    noisy_img .= xor.(noisy_img, rand(10, 10) .< 0.2)
    energy_fn = I_new -> energy(I_new, noisy_img)
    result = simulated_annealing(noisy_img, energy_fn, default_neighbor; max_iter=10)
    @test typeof(result) == typeof(noisy_img)
    @test size(result) == size(noisy_img)
end

# Test that default_neighbor changes the state
@testset "default_neighbor changes state" begin
    using ..utils
    img = falses(5, 5)
    img2 = default_neighbor(img)
    @test img != img2
end

# Test that parameter optimization returns a NamedTuple with expected fields
@testset "parameter optimization output" begin
    using ..utils
    img = falses(10, 10)
    img[3:7, 3:7] .= true
    noisy_img = copy(img)
    noisy_img .= xor.(noisy_img, rand(10, 10) .< 0.2)
    initial_params = (alpha=1.0, beta=2.0, T_init=2.0, cooling=0.99, max_iter=10)
    param_energy_fn = p -> param_energy(p, img, noisy_img)
    best = simulated_annealing(initial_params, param_energy_fn, neighbor_params; max_iter=2)
    @test haskey(best, :alpha) && haskey(best, :beta) && haskey(best, :T_init) && haskey(best, :cooling) && haskey(best, :max_iter)
end
