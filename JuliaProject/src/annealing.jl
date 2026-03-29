# My simulated annealing implementation
module annealing

export simulated_annealing, simulated_annealing_log

# Simulated annealing algorithm
function simulated_annealing(
    state,
    energy_fn::Function,
    neighbor_fn::Function;
    T_init::Float64 = 2.0,
    T_min::Float64 = 1e-4,
    cooling::Float64 = 0.99,
    max_iter::Int = 100_000
)
    I = state
    T = T_init
    E = energy_fn(I)
    for _ in 1:max_iter
        if T < T_min
            break
        end
        I_new = neighbor_fn(I)
        E_new = energy_fn(I_new)
        dE = E_new - E
        if dE < 0 || rand() < exp(-dE / T)
            I = I_new
            E = E_new
        end
        T *= cooling
    end
    return I
end

# Simulated annealing with logging of best energy/parameters
function simulated_annealing_log(
    state,
    energy_fn::Function,
    neighbor_fn::Function;
    T_init::Float64 = 2.0,
    T_min::Float64 = 1e-4,
    cooling::Float64 = 0.99,
    max_iter::Int = 100_000,
    log_interval::Int = 10
)
    I = state
    T = T_init
    E = energy_fn(I)
    best_I = I
    best_E = E
    log = [(0, best_E)]
    for iter in 1:max_iter
        if T < T_min
            break
        end
        I_new = neighbor_fn(I)
        E_new = energy_fn(I_new)
        dE = E_new - E
        if dE < 0 || rand() < exp(-dE / T)
            I = I_new
            E = E_new
        end
        if E < best_E
            best_E = E
            best_I = deepcopy(I)
        end
        if iter % log_interval == 0
            push!(log, (iter, best_E))
        end
        T *= cooling
    end
    return best_I, log
end

end
