module AMP

include("../common.jl")

predict(x, A) = abs.(A'x)

loss(x, A, y) = MSE(predict(x, A), y)

function AMP_βinf(problem;
            seed = -1,
            lr = 1,
            epochs = 200,
            infotime = 1,  # report every `infotime` epochs
            verb = 2,
            μ = 0.1,
            ψ = 0.,
            ρ₀ = 0.,
            reinf = 0.,
            β = 100,
            backtrack=false, # lower the lr if neeeded to have non increasing steps
            ϵ = 1e-10)

    @extract problem: A y Atst ytst teacher


    seed > 0 && srand(seed)
    N, M = size(A)
    verb > 1 && info("# N=$N, M=$M, α=$(M/N)")
    @assert length(y) == M


    α = M/N
    fnA2 = vecnorm(A)^2
    A2 = A.^2

    x = (1-ρ₀) .* randn(N) .+ ρ₀.*teacher.x0

    l = loss(x, A, y)
    println("epoch:0   loss:", l, "     ρ:", overlap(x, teacher[2]))
    # τᵣ = ones(N)
    # τₓ = ones(N)
    # τₚ = ones(M)
    τᵣ = 1
    τₓ = 1 #/(μ+ϵ)
    τₚ = 1
    p = A'x
    s = zeros(M)
    # τₛ = ones(M)
    τₛ = 1
    r = zeros(N)


    for t = 1:epochs
        ### AMP-MaxSum Equations from Rangan's paper
        ### Seems to work fine if (from best to worse):
        ### - (until α = 1.5, maybe) Consider the actual MaxSum equations but force τₛ to become small (reinf ~ 1e-3 / 1e-4 or less for small α), adn small μ ~ 1e-5
        ### - (below α = 2.5) Consider the actual MaxSum equations but force τₛ to become small (reinf = 1e-3 / 1e-4 or less for small α), μ = 0
        ### - (at least up to α = 2.5) Consider the smoothened version with β ~ 10, reinf = 0, μ = 0
        ###
        ### ψ = 0.3 seems to be a good value for damping

        ### possible annealng in β
        # β += 1e-2

        τₚ_new = (1/M * fnA2 * τₓ) ### same damping scheme used in Florent's code
        p .= (A'x - τₚ_new*s) * (1-ψ) + p * ψ
        τₚ = τₚ_new * (1-ψ) + τₚ * ψ

        ### actual argmax (β->∞)
        zs = (2τₚ.*y .+ abs.(p)) ./ (1 + 2τₚ) .* sign.(p)
        minus_∂g = (-2) / ((-2)*τₚ - 1) ### actual derivative (β->∞), but ignoring the Dirac δ(zs)
        ### smoothened version of the argmax
        # zs = (2τₚ.*y .* tanh.(β * p) .+ abs.(p) .* tanh.(β * p)) ./ (1 + 2τₚ)
        # f2 = @. -2 * ((β*(1-tanh.(β*zs)^2).*zs + tanh.(β*zs))^2 + (y .-tanh.(β*zs).*zs)*(-β*(1-tanh.(β*zs)^2)+2β^2*zs .* tanh.(β*zs)*(1-tanh.(β*zs)^2)-β*(1-tanh.(β*zs)^2)))
        # minus_∂g = mean(f2 ./ (τₚ .* f2 .- 1))

        s .= (zs .- p) ./ τₚ #* (1-ψ) + s * ψ  ### possible damping
        τₛ = minus_∂g * (1-tanh(t*reinf))  ### same as Rangan when reinf=0.
        τᵣ = ((1/N * fnA2 * τₛ)^(-1))
        r .= (x .+ τᵣ .* (A * s)) #* (1-ψ) + r * ψ ### possible damping
        x .= (r .* 1 ./ (1 + τᵣ*μ)) #* (1-ψ) + x * ψ ### possible damping
        τₓ = τᵣ / (1 + τᵣ*μ)


        #############################
        ######## other stuff ########
        #############################
        # s .= (2 ./ (2τₚ + 1) .* (y .* sign.(p) .- p)) #* (1-ψ) + s * ψ
        # # AMP.A
        # τₚ = 1/α * ((1-tanh(t*reinf))^(-1)
        # p .= (A'x - τₚ / mag * g) * mag / (mag + τₓ*μ)
        # g .= (y .* sign.(p) .- p)
        # m∂g = 2*(mag + τₓ*μ) / (2τₚ + (mag + τₓ*μ))
        # τₓ = ((1/N * fnA2 * (1-tanh(t*reinf)))^(-1))
        # x .= (x * mag / (mag + τₓ*μ) .+ τₓ .* (A * g))

        # τₚ = (1/M * fnA2 * τₓ) * (1-ψ) + τₚ * ψ
        # p .= (A'x - 2/α .* s) * (1-ψ) + p * ψ
        # s .= (2 ./ (1 .+ 2τₚ) .* (y .* sign.(p) .- p))
        # ∂g = (-mean(sign.(p) ./ (τₚ .* sign.(p) .- 1)))
        # x .= 2*(-∂g .* 2 ./ (1 .+ 2τₚ) .* x + A * s)  * (1-ψ) + x * ψ
        ρ = overlap(x, teacher[2])
        if abs(ρ) > 1-ϵ || t % infotime == 0
            l = loss(x, A, y)
            println("\r epoch:$t   loss:", l, "     ρ:", overlap(x, teacher[2]), " norm:", vecnorm(x))
            @show τᵣ, τₓ, τₚ, τₛ
            abs(ρ) > 1-ϵ && break
        end
    end
    println()

    return
end

end #module
