module ASP
using ExtractMacro
using ForwardDiff
using Parameters

include("../common.jl")

struct ASPGraph
    x::Vector{Float64}
    Δ0::Vector{Float64}
    Δ1::Vector{Float64}
    B::Vector{Float64}
    A0::Vector{Float64}
    A1::Vector{Float64}

    ω::Vector{Float64}
    V0::Vector{Float64}
    V1::Vector{Float64}
    g::Vector{Float64}
    Γ0::Vector{Float64}
    Γ1::Vector{Float64}

    y::Vector{Float64}
    W::Matrix{Float64}
    W2::Matrix{Float64}
end

function ASPGraph(problem)
    @extract problem: W=A y
    N, M = size(W)
    @assert length(y) == M

    ASPGraph( [zeros(N) for _=1:6]...,
        [zeros(M) for _=1:6]...,
        y, W, W.^2)
end


function Base.show(io::IO, prob::ASPGraph)
    print(io, "ASPGraph:\n")
    for f in fieldnames(ASPGraph)
        print(io, "  $f: $(summary(getfield(prob, f)))\n")
    end
end


function init!(asp::ASPGraph, prms)
    asp.x .= initx(asp.W, asp.y, prms)
    asp.Δ0 .= 2
    asp.Δ1 .= 1
    asp.g .= 0
end

predict(x, A) = abs.(A'x)
loss(x, A, y) = MSE(predict(x, A), y)

function initx(A, y, prms)
    @extract prms: ρ₀ x₀
    N = size(A, 1)
    if ρ₀ >= 0 && x₀ == :teacher
        x₀ = (1-ρ₀) .* zeros(N) .+ ρ₀ .* prms.teacher.x0
    end
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
    x
end

#Mathematica Compatibility
Power(x,a) = x^a

### Ge and derivatives from replica calculations
function Ge₀(y, h, q10, δq, m)
    a = sqrt((1 + 2*δq)*(1 + 2*m*q10 + 2*δq))
    # @assert abs(-((2*m*q10*y - h*(1 + 2*δq))/(sqrt(q10)*a))) < 45
    Z1 = H(-((2*m*q10*y - h*(1 + 2*δq))/(sqrt(q10)*a))) * exp(-(m*Power(h + y,2))/(1 + 2*m*q10 + 2*δq))
    Z2 = H(-((2*m*q10*y + h*(1 + 2*δq))/(sqrt(q10)*a)))*exp(-(m*Power(h - y,2))/(1 + 2*m*q10 + 2*δq))
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

# function ∂h_Ge₀(y, h, q10, δq, m)
#     return (Ge₀(y, h+1e-6, q10, δq, m) - Ge₀(y, h-1e-6, q10, δq, m))/2e-6
# end
#
# function ∂δq_Ge₀(y, h, q10, δq, m)
#     return (Ge₀(y, h, q10, δq+1e-6, m) - Ge₀(y, h, q10, δq-1e-6, m))/2e-6
# end
#
# function ∂q10_Ge₀(y, h, q10, δq, m)
#     return (Ge₀(y, h, q10+1e-6, δq, m) - Ge₀(y, h, q10-1e-6, δq, m))/2e-6
# end
###############################

## use derivative from replica computations
∂ω_ϕ(y, ω, V0, V1, m) = ∂h_Ge₀(y, ω, V0, V1, m)
∂V0_ϕ(y, ω, V0, V1, m) = ∂q10_Ge₀(y, ω, V0, V1, m)
∂V1_ϕ(y, ω, V0, V1, m) =  ∂δq_Ge₀(y, ω, V0, V1, m)

function oneiter!(asp::ASPGraph, t, prms)
    @extract asp: x Δ0 Δ1 A0 A1 B ω V0 V1 g Γ0 Γ1 y W W2
    @extract prms: m λ
    α = size(W, 2) / size(W, 1)


    γ = tanh(t*1e-4)
    # Bold = copy(B)
    ψ = 0.5

    V0new =  W2'*Δ0
    V1new = W2'*Δ1
    ω .= (W' * x .-  g .* (m.*V0new + V1new)) .* (1-ψ) .+ ω .* ψ
    V0 .= V0new .* (1-ψ) .+ V0 .* ψ
    V1 .= V1new .* (1-ψ) .+ V1 .* ψ


    @. g = ∂ω_ϕ(y, ω, V0, V1, m)
    @. Γ0 = 2∂V1_ϕ(y, ω, V0, V1, m) - g^2
    @. Γ1 = -2∂V0_ϕ(y, ω, V0, V1, m) + m*g^2 + m*Γ0

    A0 .= W2*Γ0
    A1 .= W2*Γ1
    # @assert all(x->x>=0, A0)
    # @assert all(A1 .+ λ .- m.*A0 .> 0)

    @. A1 = max(A1, -λ -γ + m*A0 + 1e-15)
    B .= W*g .- x.*(m.*A0 .- A1)

    # @. xnew = ∂B_ϕh(B, A0, A1+λ, m)
    # @. Δ0 = -∂A1_ϕh(B, A0, A1+λ, m) - xnew^2
    # @. Δ1 = ∂²B_ϕh(B, A0, A1+λ, m) - m*Δ0

    xnew  = @. B / (A1 + λ - m*A0)
    @. Δ0 = A0 / (A1+λ) / (A1 + λ - m*A0)
    # @. Δ1 = 1 / (A1 + λ) #1 / (A1 + λ - m*A0) - m*Δ0
    @. Δ1 = 1 / (A1 + λ - m*A0) - m*Δ0

    # xnew  = @. (B + γ*Bold) / (A1 + λ + γ - m*A0)
    # @. Δ0 = A0 / (A1 + λ + γ) / (A1 + λ + γ - m*A0)
    # @. Δ1 = 1 / (A1 + λ + γ) #1 / (A1 + λ - m*A0) - m*Δ0


    # @assert all(x->x>=0,Δ0)
    # @assert all(x->x>=0,Δ1)

    Δ = norm(x.-xnew) / length(x)
    x .= xnew
    Δ
end

## Parameters for solve
@with_kw mutable struct Params
    λ=0. # L2 reg
    dropλ = true # set λ=0 after convergence
    seed = -1
    epochs = 200
    infotime = 1  # report every `infotime` epochs
    x₀ = :randn   # initial configuration
                  # [:spectral, :teacher,  :randn, nothing, a configuration]
    ρ₀ = 1e-8     # initial overlap with teacher if x₀==:teacher
    verb = 3
    m = 1  # parisi parameter
    ϵ = 1e-8  # stopping criterion
end

function solve(; N=1000,α=1.5, seedp=-1, bias=0., R=1., kws...)
    prob = Problem("gle", N=N, α=α, act=abs, bias=bias, R=R, seed=seedp)
    solve(prob; kws...)
end
solve(problem; kws...) = solve(problem, Params(;kws...))

function solve(problem, prms::Params)
    @extract problem: A y Atst ytst teacher

    prms.seed > 0 && srand(prms.seed)
    N, M = size(A)
    @assert length(y) == M

    ## printing utilities ##
    df = DataFrame(epoch = Int[],
            train_loss = Float64[], test_loss = Float64[],
            ρ = Float64[], xnorm=Float64[])

    report(epoch, Δ, verb) = begin
            x = asp.x
            res = (epoch=epoch,
                    train_loss = loss(x, A, y),
                    test_loss = loss(x, Atst, ytst),
                    # ρ = dot(x, teacher.x0)/length(x),
                    ρ = dot(x, teacher.x0)/sqrt(dot(x,x)*dot(teacher.x0,teacher.x0)),
                    xnorm = sqrt(dot(x, x)/length(x)))

            push!(df, res)
            verb > 1 &&  (cprint(res); cprintln(:Δ=>Δ));
        end
    ########################

    asp = ASPGraph(problem)
    init!(asp, prms)
    ok = false
    report(0, 1, prms.verb)
    for epoch=1:prms.epochs
        Δ = oneiter!(asp, epoch, prms)
        epoch % prms.infotime == 0 && report(epoch, Δ, prms.verb)
        ok = Δ < prms.ϵ
        # prms.λ < 0 && break
        # if epoch % 50 == 0
        #     prms.m += 200
        #     prms.λ -= 0.00001
        #     println("set λ=$(prms.λ) m=$(prms.m)")
        # end
        if ok
            (prms.λ <= 0 || !prms.dropλ) && break
            prms.verb > 0 && println("# set λ=0")
            prms.λ = 0
        end
    end
    prms.verb > 0 && !ok && @warn("not converged!")
    return df, asp, ok
end

end #module
