module ASP
using ExtractMacro
using ForwardDiff
using Parameters
using TimerOutputs

include("common.jl")

struct AMPGraph
    x::Vector{Float64}
    Δ0::Vector{Float64}
    Δ1::Vector{Float64}
    B::Vector{Float64}

    ω::Vector{Float64}
    g::Vector{Float64}
    Γ0::Vector{Float64}
    Γ1::Vector{Float64}
    
    y::Vector{Float64}
    W::Matrix{Float64}
    ΔW::Float64 # empirical E[w^2_ij] 
end

function AMPGraph(problem)
    @extract problem: W=A y
    N, M = size(W)
    @assert length(y) == M
    
    AMPGraph( [zeros(N) for _=1:4]...,
        [zeros(M) for _=1:4]...,
        y, W, vecnorm(W)^2/(M*N))
end

function init!(amp::AMPGraph, prms, problem=nothing)
    amp.x .= initx(amp.W, amp.y, prms, problem)
    amp.Δ0 .= 1
    amp.Δ1 .= 1
    amp.g .= 0
end

predict(x, A) = abs.(A'x)
loss(x, A, y) = MSE(predict(x, A), y)

function initx(A, y, prms, problem=nothing)
    @extract prms: ρ₀ x₀
    N = size(A, 1)  
    if x₀ isa Vector
        x = deepcopy(x₀)
    elseif x₀ == :spectral
        x = spectral_init_optimal(A, y)
    elseif x₀ == :randn
        x = randn(N)
    elseif x₀ == nothing
        x = zeros(N)
    end
    @assert length(x) == N
    x .= x .+ ρ₀ .* problem.teacher.x0
    x
end

#Mathematica Compatibility
Power(x,a) = x^a

### Ge and derivatives from replica calculations
function Ge₀(y, h, q10, δq, m)
    a = sqrt((1 + 2*δq)*(1 + 2*m*q10 + 2*δq))
    Z1 = H(-((2*m*q10*y - h*(1 + 2*δq))/
        (sqrt(q10)*a)))*exp(-(m*Power(h + y,2))/(1 + 2*m*q10 + 2*δq))
    Z2 = H(-((2*m*q10*y + h*(1 + 2*δq))/
        (sqrt(q10)*a)))*exp(-(m*Power(h - y,2))/(1 + 2*m*q10 + 2*δq))
    Z = (1 + 2*δq)*(Z1 + Z2) / a
    1/m * log(Z)
end

∂h_Ge₀ = (y, h, q10, δq, m) -> begin
            ForwardDiff.derivative(h->Ge₀(y, h, q10, δq, m),h)
        end
 ∂δq_Ge₀ = (y, h, q10, δq, m) -> begin
            ForwardDiff.derivative(δq->Ge₀(y, h, q10, δq, m),δq)
        end
∂q10_Ge₀ = (y, h, q10, δq, m) -> begin
            ForwardDiff.derivative(q10->Ge₀(y, h, q10, δq, m),q10)
        end
###############################

## use derivative from replica computations
∂ω_ϕ(y, ω, V0, V1, m) = ∂h_Ge₀(y, ω, V0, V1, m)
∂V0_ϕ(y, ω, V0, V1, m) = ∂q10_Ge₀(y, ω, V0, V1, m) 
∂V1_ϕ(y, ω, V0, V1, m) =  ∂δq_Ge₀(y, ω, V0, V1, m)

# @timeit function updV(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
#     # V0 .= W2'*Δ0
#     # V1 .= W2'*Δ1
#     V0 .= sum(Δ0) / length(Δ0) # assume E[wij^2]=1/N
#     V1 .= sum(Δ1) / length(Δ0) # assume E[wij^2]=1/N 
#     @. V0 = max(V0, 1e-12)
#     @. V1 = max(V1, 1e-12)
# end

# @timeit function updω(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
#     ω .= W' * x .-  g .* (m.*V0 + V1) 
# end

# @timeit function updg(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
#     @. g = ∂ω_ϕ(y, ω, V0, V1, m)
#     @. Γ0 = 2∂V1_ϕ(y, ω, V0, V1, m) - g^2
#     @. Γ1 = -2∂V0_ϕ(y, ω, V0, V1, m) + m*g^2 + m*Γ0
# end


# @timeit function updA(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
#     A0 .= sum(Γ0) / length(Δ0) # assume E[wij^2]=1/N
#     A1 .= sum(Γ1) / length(Δ0) # assume E[wij^2]=1/N  
#     # @assert all(x->x>=0, A0)
#     # @assert all(A1 .+ λ .- m.*A0 .> 0)
#     @. A0 = max(A0, 1e-12)
#     @. A1 = max(A1, -λ + m*A0 + 1e-12)
# end

# @timeit function updx(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
#     xnew  = @. B / (A1 + λ - m*A0)
#     @. Δ0 = A0 / (A1+λ) /(A1 + λ - m*A0)
#     @. Δ1 = 1 / (A1 + λ - m*A0) - m*Δ0 
    
#     # @assert all(x->x>=0,Δ0)
#     # @assert all(x->x>=0,Δ1)
    
#     Δ = norm(x.-xnew) / length(x)
#     x .= xnew 
#     Δ
# end

# @timeit function updB(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
#     B .= W*g .- x.*(m.*A0 .- A1)
# end

@timeit function oneiter!(amp::AMPGraph, t, prms)
    @extract amp: x Δ0 Δ1 B ω  g Γ0 Γ1 y W #W2
    @extract prms: m λ
    N, M = size(W) 
    α = M / N
    
    # updV(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
    # updω(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
    # updg(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
    # updA(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
    # updB(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W, W2,m, λ)
    # Δ=updx(x,Δ0, Δ1, A0, A1, B, ω, V0, V1, g, Γ0, Γ1, y, W,W2, m, λ)
    # V0 = W2'*Δ0
    # V1 = W2'*Δ1
    V0 = sum(Δ0) / N # assume E[wij^2]=1/N
    V1 = sum(Δ1) / N # assume E[wij^2]=1/N 
    V0 = max(V0, 1e-12)
    V1 = max(V1, 1e-12) 
    ω .= W' * x .-  g .* (m.*V0 + V1) 
    
    @. g = ∂ω_ϕ(y, ω, V0, V1, m)
    @. Γ0 = 2∂V1_ϕ(y, ω, V0, V1, m) - g^2
    @. Γ1 = -2∂V0_ϕ(y, ω, V0, V1, m) + m*g^2 + m*Γ0
    
    # A0 .= W2*Γ0
    # A1 .= W2*Γ1 
    A0 = sum(Γ0) / N # assume E[wij^2]=1/N
    A1 = sum(Γ1) / N # assume E[wij^2]=1/N  
    # @assert all(x->x>=0, A0)
    # @assert all(A1 .+ λ .- m.*A0 .> 0)
    A0 = max(A0, 1e-12)
    A1 = max(A1, -λ + m*A0 + 1e-12)
    B .= W*g .- x.*(m.*A0 .- A1)
    
    # @. xnew = ∂B_ϕh(B, A0, A1+λ, m)
    # @. Δ0 = -∂A1_ϕh(B, A0, A1+λ, m) - xnew^2
    # @. Δ1 = ∂²B_ϕh(B, A0, A1+λ, m) - m*Δ0
    xnew  = @. B / (A1 + λ - m*A0)
    @. Δ0 = A0 / (A1+λ) /(A1 + λ - m*A0)
    @. Δ1 = 1 / (A1 + λ - m*A0) - m*Δ0 
    # @assert all(x->x>=0,Δ0)
    # @assert all(x->x>=0,Δ1)
    
    Δ = norm(x.-xnew) / length(x)
    x .= xnew 
    Δ
end

## Parameters for solve
@with_kw mutable struct Params
    λ::Float64 = 0. # L2 reg
    dropλ::Bool = true # set λ=0 after convergence   
    seed::Int = -1
    epochs::Int = 200
    infotime::Int = 1  # report every `infotime` epochs
    x₀ = :randn   # initial configuration 
                  # [:spectral, :teacher,  :randn, nothing, a configuration]
    ρ₀::Float64 = 0     # initial overlap with teacher if x₀==:teacher
    verb::Int = 3 
    m::Float64 = 1  # parisi parameter
    ϵ::Float64 = 1e-8  # stopping criterion
end

solve(; N=1000,α=1.5, seedp=-1, kws...) = solve(Problem("gle",N=N,α=α,act=abs,seed=seedp); kws...)
solve(problem; kws...) = solve(problem, Params(;kws...))

function solve(problem, prms::Params)
    @extract problem: A y Atst ytst teacher

    prms.seed > 0 && srand(prms.seed)
    N, M = size(A)
    @assert length(y) == M

    ## printing utilities ##
    df = DataFrame(epoch=Int[], m=Float64[], λ=Float64[], 
            train_loss=Float64[], test_loss=Float64[],
            ρ=Float64[], xnorm=Float64[])

    report(epoch, Δ, verb) = begin
            x = amp.x
            res = (epoch=epoch,
                    m=prms.m, λ=prms.λ,
                    train_loss = loss(x, A, y),
                    test_loss = loss(x, Atst, ytst),
                    ρ = dot(x, teacher.x0)/length(x),
                    xnorm = sqrt(dot(x, x)/length(x)))

            push!(df, res)
            verb > 1 &&  (cprint(res); cprintln(:Δ=>Δ));
        end
    ########################

    amp = AMPGraph(problem)
    init!(amp, prms, problem)
    ok = false
    report(0, 1, prms.verb)
    for epoch=1:prms.epochs
        Δ = oneiter!(amp, epoch, prms)
        epoch % prms.infotime == 0 && report(epoch, Δ, prms.verb)
        ok = Δ < prms.ϵ
        if ok
            (prms.λ == 0 || !prms.dropλ) && break
            prms.verb > 0 && println("# set λ=0") 
            prms.λ = 0
        end
    end
    prms.verb > 0 && !ok && warn("not converged!")
    return df, amp, prms, ok
end

end #module
