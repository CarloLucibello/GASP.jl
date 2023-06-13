include("../../src/common.jl") 
include("../../src/solvers/ASP-UNSAT.jl")

# Define the problem
N = 100
α = 8.0
seed = 13
prob = ASP.Problem("gle"; act=abs, N, α, seed)

# Solve the problem
df, asp, ok = ASP.solve(prob, verb=1, epochs=100)
df 