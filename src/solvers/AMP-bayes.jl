
module AMP
using ExtractMacro
include("../common.jl")
myact = abs

struct AMPGraph
    x::Vector{Float64}
    σ::Vector{Float64}
    ω::Vector{Float64}
    V::Vector{Float64}
    g::Vector{Float64}
    ∂g::Vector{Float64}
    A::Vector{Float64}
    B::Vector{Float64}
    
    λ::Float64 # L2 regularizer
    
    y::Vector{Float64}
    W::Matrix{Float64}
    W2::Matrix{Float64}
end

function AMPGraph(problem; λ=0.)
    @extract problem: W=A y
    N, M = size(W)
    @assert length(y) == M
    x,σ,A,B = zeros(N),zeros(N),zeros(N),zeros(N)
    ω,V,g,∂g = zeros(M),zeros(M),zeros(M),zeros(M)
    
    AMPGraph(x, σ, ω, V,
        g, ∂g, A, B,
        λ, 
        y, W, W.^2)
end

function init!(amp::AMPGraph; x₀=nothing)
    amp.x .= initx(x₀, amp.W, amp.y)
    amp.σ .= 1
    amp.g .= 0
    amp.∂g .= 0
    amp.ω .= amp.y
    amp.V .=  amp.W2' * amp.σ
end

predictor(amp::AMPGraph) = amp.x

predict(x, A) = myact.(A'x)
loss(x, A, y) = MSE(predict(x, A), y)


function δ(x, ϵ=1e-4)
    x = abs(x) / ϵ
    ifelse(x > 1, 0., 1/ ϵ * (1 - x)) 
end

function oneiter!(amp::AMPGraph)
    @extract amp: x σ B A  
    @extract amp: ω g ∂g V 
    @extract amp: y W W2 λ
    # N, M = size(W)
    # α = M/N
    ϵ = 1e-10 # for stability
    V .= W2' * σ #|> mean
    ω .= W' * x .- V .* g

    kappa = 1e-10
    @. g = ifelse(y < kappa,
        (y - ω)/(V+ϵ),
        (-ω + y*tanh(y*ω/(V+ϵ))) / (V+ϵ))

    @. ∂g = -1/(V+ϵ) + (1 - tanh(y*ω/(V+ϵ))^2)*(y/(V+ϵ))^2
    # ∂g .= mean(∂g) 
    # ∂g .= -mean(g.^2) #equivalent to above? 

    A .= .- W2*∂g #|> mean
    A .= max.(A, -λ+ϵ)
    # @assert all(A .+ λ .> 0) 
    B .= W*g .+ A.*x

    xnew  = @. B  / (A + λ)
    Δ = maximum(abs, x .- xnew) 
    x .= xnew
    @. σ = 1 / (A + λ)

    Δ
end

function oneiter_swamp!(amp::AMPGraph, t)
    @extract amp: x σ B A  
    @extract amp: ω g ∂g V 
    @extract amp: y W W2 λ
    # N, M = size(W)
    # α = M/N
    ϵ = 1e-10 # for stability
    kappa = 1e-3
    if t > 1
        @. g = ifelse(y < kappa,
                    (y - ω) / (V+ϵ),
                    (-ω + y*tanh(y*ω/(V+ϵ))) / (V+ϵ))
    else
        @. g = 0
    end
    V .= W2' * σ #|> mean
    ω .= W' * x .- V .* g

    this_g = @. ifelse(y < kappa,
        (y - ω)/(V+ϵ),
        (-ω + y*tanh(y*ω/(V+ϵ))) / (V+ϵ))

    @. ∂g = -1/(V+ϵ) + (1 - tanh(y*ω/(V+ϵ))^2)*(y/(V+ϵ))^2
    # ∂g .= mean(∂g) 
    # ∂g .= -mean(this_g.^2) #equivalent to above? 

    A .= .- W2*∂g #|> mean
    A .= max.(A, -λ+ϵ)
    B .= W*this_g .+ A.*x

    xnew  = @. B  / (A + λ)
    @. σ = 1 / (A + λ)

    Δ = maximum(abs, x .- xnew) 
    x .= xnew
    Δ
end

function initx(x₀, A, y)
    N = size(A,1)
    if x₀ isa Vector
        x = deepcopy(x₀)
    elseif x₀ == :spectral
        x = spectral_init_optimal(A, y)
    elseif x₀ == :randn
        x = randn(N) / sqrt(N)
    elseif x₀ == nothing
        x = zeros(N)
    end
    @assert length(x) == size(A, 1)
    x
end

function solve(; N=1000,α=1.5, seedp=-1, kws...)
    prob = Problem("gle", N=N, α=α, act=myact, seed=seedp)
    solve(prob; kws...)
end

function solve(problem;
            λ=1., # L2 reg
            seed = -1,
            epochs = 200,
            infotime = 1,  # report every `infotime` epochs
            x₀ = nothing, # [:spectral,:techer, nothing, a configuration]
            ρ₀ = 1e-5, # initial overlap with the teacher (if x₀=:teacher)
            verb = 3,
            swamp = true,
            ϵ = 1e-7  # stopping criterion
        )

    @extract problem: A y Atst ytst teacher
    N, M = size(A)
    seed > 0 && srand(seed)
    amp = AMPGraph(problem, λ=λ)


    df = DataFrame(epoch = Int[],
            train_loss = Float64[],
            test_loss = Float64[],
            ρ = Float64[],
            xnorm=Float64[])

    report(epoch, Δ, verb) = begin
            x = predictor(amp)
            res = (epoch=epoch,
                    train_loss = loss(x, A, y),
                    test_loss = loss(x, Atst, ytst),
                    ρ = abs(dot(x, teacher.x0)/length(x)),
                    xnorm = sqrt(dot(x, x)/length(x)))

            push!(df, res)
            verb > 1 &&  (cprint(res); cprintln(:Δ=>Δ));
        end

    epoch = 0
    Δ = 1.
    if ρ₀ >= 0 && x₀ == :teacher
        x₀ = (1-ρ₀) .* zeros(N) .+ ρ₀.*teacher.x0
    end
    init!(amp, x₀=x₀)
    # Δ = oneiter!(amp; first=true)
    report(epoch, Δ, verb+1);# try
    while epoch < epochs
        epoch += 1
        if swamp
            Δ = oneiter_swamp!(amp, epoch)
        else
            Δ = oneiter!(amp)
        end
        epoch % infotime == 0 && report(epoch, Δ, verb)
        Δ < ϵ && break
    end
    #catch e; e isa InterruptException || error(e); end
    # report(epoch, Δ, verb+1)
    verb > 0 && Δ > ϵ && @warn("not converged!")

    return problem, amp, predictor(amp)
end

end #module
