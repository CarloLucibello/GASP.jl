
module AMP
using ExtractMacro

include("../common.jl")

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
end

predictor(amp::AMPGraph) = amp.x

predict(x, A) = abs.(A'x)
loss(x, A, y) = MSE(predict(x, A), y)

function δ(x, ϵ=1e-2)
    x = abs(x) / ϵ
    ifelse(x > 1, 0., 1/ ϵ * (1 - x)) 
end

function oneiter!(amp::AMPGraph)
    @extract amp: x σ A B  
    @extract amp: ω V g ∂g
    @extract amp: y W W2 λ
    # V .= W2' * σ
    # ω .= W' * x .- V .* g
    
    # @. g = 2/(1+2V) * (y * sign(ω) - ω)
    # @. ∂g = 2/(1+2V) * (2y*δ(ω) - 1)
    
    # A .= .- W2*∂g
    # B .= W*g .+ A.*x
    
    # xnew = @. B / (A + λ)
    # @. σ = 1 / (A + λ)
    
    α = size(W, 2) / size(W, 1)
    ω .= W'*x #.- (y .* sign.(ω) .- ω) ./ α 
    divg = mean(@. 2y*δ(ω) - 1)/2
    xnew = W*(y .* sign.(ω) .- ω) #-divg .* x
    Δ = norm(x.-xnew) / length(x)
    x .= xnew
    
    Δ
end

function initx(x₀, A, y)
    if x₀ isa Vector
        x = deepcopy(x₀)
    elseif x₀ == :spectral
        x = spectral_init_optimal(A, y)
    elseif x₀ == nothing
        # x = zeros(size(A,1))
        x = randn(size(A,1))
    end
    @assert length(x) == size(A, 1)
    x
end

function solve(; N=1000,α=1.5, seedp=-1, kws...)
    prob = Problem("gle", N=N, α=α, act=abs, seed=seedp)
    solve(prob; kws...)
end

function solve(problem;
            λ=0., # L2 reg
            seed = -1,
            epochs = 200,
            infotime = 1,  # report every `infotime` epochs
            x₀ = nothing, # [:spectral, nothing, a configuration]
            verb = 3,
            ϵ = 1e-10  # stopping criterion
        )

    @extract problem: A y Atst ytst teacher

    seed > 0 && srand(seed)
    amp = AMPGraph(problem, λ=λ)

    df = DataFrame(epoch = Int[],
            train_loss = Float64[],
            test_loss = Float64[],
            ρ = Float64[],
            xnorm=Float64[])

    report(epoch, Δ, verb) = begin
            x = predictor(amp)
            res = @NT(epoch=epoch,
                    train_loss = loss(x, A, y),
                    test_loss = loss(x, Atst, ytst),
                    ρ = abs(dot(x, teacher.x0)/length(x)),
                    xnorm = sqrt(dot(x, x)/length(x)))

            push!(df, res)
            verb > 1 &&  (cprint(res); cprintln(:Δ=>Δ));
        end

    epoch = 0
    Δ = 1.
    init!(amp, x₀=x₀)
    report(epoch, Δ, verb+1);# try
    while epoch < epochs
        epoch += 1
        Δ = oneiter!(amp)
        epoch % infotime == 0 && report(epoch, Δ, verb)
        Δ < ϵ && break
    end
    #catch e; e isa InterruptException || error(e); end
    # report(epoch, Δ, verb+1)
    verb > 0 && Δ > ϵ && warn("not converged!")

    return problem, amp, predictor(amp)
end

end #module
