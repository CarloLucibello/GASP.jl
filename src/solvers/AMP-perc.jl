module Perc
using ExtractMacro

include("../common.jl")

struct FactorGraphTAP
    ξ::Matrix{F}
    ξ2::Matrix{F}
    σ::Vector{F}
    gW::Vector{F}
    ∂gW::Vector{F}
    AW::Vector{F}
    BW::Vector{F}
    λ::F #L2 regularizer
  
    function FactorGraphTAP(ξ::Matrix, σ::Vector, λ=1)
        N, M = size(ξ)
        new(ξ, ξ.^2, σ,
            zeros(M), zeros(M), 
            zeros(N), zeros(N),
            λ)
    end
end

mutable struct ReinfParams
    r::F
    rstep::F
    wait_count::Int
    ReinfParams(r=0., rstep=0.) = new(r, rstep, 0)
end

function init!(G::FactorGraphTAP)
    @extract G: AW BW gW
    BW .= 0
    AW .= 1
    gW .= 0
end

function oneBPiter!(G::FactorGraphTAP, r = 0.)
    @extract G: ξ ξ2 gW ∂gW AW BW λ σ
    
    mW = @. BW / AW
    ρW = @. 1 / AW

    V = ξ2' * ρW
    ω = ξ' * mW  .- gW .* V
    
    @. gW = σ / √V * GH(-σ * ω / √V)  
    @. ∂gW = -ω/V *gW - gW^2
    
    AW .=  -ξ2 * ∂gW 
    BW .=  ξ * gW + AW .* mW + r .* BW
    
    AW .+= λ + r .* AW
    Δ = mean(abs, BW ./ AW .- mW)
    Δ
end

function update_reinforcement!(reinfpar::ReinfParams)
    if reinfpar.wait_count < 10
        reinfpar.wait_count += 1
    else
        reinfpar.r = 1 - (1-reinfpar.r) * (1-reinfpar.rstep)
    end
end

function converge!(G::FactorGraphTAP; maxiters = 10000, ϵ=1e-5
                                , altsolv=false, altconv = false
                                , reinfpar::ReinfParams=ReinfParams()
                                , xteacher=zeros(0))

    println("ρ=", dot(magsW(G), xteacher)/length(xteacher))
                                    
    for it=1:maxiters
        Δ = oneBPiter!(G, reinfpar.r)
        E = 0
        # E = errors(G.ξ,sign.(G.Bh), magsW(G))
        @printf("it=%d r=%.3f E(W=mags)=%d  M=%.4f M2=%.4f \tΔ=%f \n",
                it, reinfpar.r, E, mean(magsW(G)), mean(varsW(G)), Δ)
        
        println("ρ=", dot(magsW(G), xteacher)/length(xteacher))
        update_reinforcement!(reinfpar)
        if altsolv && E == 0
            println("Found Solution!")
            break
        end
        if altconv && Δ < ϵ
            println("Converged!")
            break
        end
    end   
end

magsW(G::FactorGraphTAP) = G.BW ./ G.AW 
varsW(G::FactorGraphTAP) = 1 ./ G.AW 


function solve(; N=1000,α=1.5, seedp=-1, kws...)
    prob = Problem("gle", N=N, α=α, act=sign, seed=seedp)
    solve(prob; kws...)
end

function solve(prob;
                maxiters = 10000,
                ϵ = 1e-4,
                method = :reinforcement, #[:reinforcement, :decimation]
                r = 0., rstep= 0.001,
                λ = 1., # L2 regularizer
                altsolv = false,
                altconv = true,
                seed = -1)

    @extract prob: ξ=A σ=y teacher
    seed > 0 && srand(seed)

    G = FactorGraphTAP(ξ, σ, λ)
    init!(G)

    # if method == :reinforcement
    reinfpar = ReinfParams(r, rstep)
    converge!(G, maxiters=maxiters, ϵ=ϵ, xteacher=teacher.x0, 
        reinfpar=reinfpar, altsolv=altsolv, altconv=altconv)
    return magsW(G)
end

end #module
