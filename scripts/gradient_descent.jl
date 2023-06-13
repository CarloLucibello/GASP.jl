using Plots; plotlyjs(size=(800,600))
# using StatPlots
# using LittleScienceTools.Measuring
using JLD2
include("../src/common.jl") 

include("../src/solvers/gradient_descent.jl")

function span(; samples = 20,
        epochs = 100000,
        lr = 1e-1,   #learning rate. eventually scaled down by backtracking.
        ϵ = 1e-6,  # stop criterium
        α = 0.1:0.1:2,
        N = [100,200],
        filename = "gd_lr$(lr)_abs2",
        backtrack=false
    )

    obstable = ObsTable([:N, :α])
    obs = [:train_loss, :test_loss, :ρ] 
    
    for N in N, α in α 
        row = obstable[N, α]
        for s in 1:samples
            println("# N=$N α=$α sample $s")
            prob = GD.Problem("gle", act=abs2, N=N, α=α, seed=s)
            
            # saving initial conditions
            prob, x, df = GD.solve(prob, x₀=:spectral, verb=0, epochs=0)
            for c in obs
                row[string(c,"₀")] &= df[end, c] 
            end

            # real run
            prob, x, df = GD.solve(prob, x₀=:spectral,lr=lr, 
                                    infotime=100, verb=1, epochs=epochs, 
                                    ϵ=ϵ, backtrack=backtrack)
            for c in obs
                row[c] &= df[end, c] 
            end

            row[:Psat] &= df[end, :train_loss] < 10*ϵ
        end
        println(obstable)

        # @save filename*".jld" obstable
        open(filename*".dat","w") do f 
            println(f, obstable)
        end
    end

    obstable
end