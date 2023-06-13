
include("../../src/common.jl") 
include("../../src/solvers/gradient_descent.jl")

# Define problem
N = 100
α = 10.0
seed = 17
prob = GD.Problem("gle"; act=abs, N, α, seed)

# Save initial condition from spectra initialization
prob, x0, df = GD.solve(prob, x₀=:spectral, verb=0, epochs=0)

# Solve the problem
prob, x, df = GD.solve(prob, x₀=x0, verb=1, epochs=2000)
