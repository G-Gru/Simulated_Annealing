# Examples

This folder contains demonstration scripts for simulated annealing in Julia.

1. denoising.jl  
   Denoise a binary image using custom simulated annealing.

2. parameter_search.jl  
   Optimize simulated annealing parameters for a single noisy image, print results and plot progress.

3. parameter_search_multi.jl  
   Optimize SA parameters over multiple noisy images, print results and plot progress.

4. parameter_search_optim.jl  
   Optimize SA parameters using Optim.jl, print results and plot progress.

5. annealing_comparison.jl  
   Compare custom vs. Optim.jl SA implementations, generates individual and combined fit plots.

All scripts write their outputs (plots/images) to the `results/` directory.