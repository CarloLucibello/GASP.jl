
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


function AMPGraph(problem, params)
    @extract problem: W=A y
    N, M = size(W)
    @assert length(y) == M
    x,σ,A,B = zeros(N),zeros(N),zeros(N),zeros(N)
    ω,V,g,∂g = zeros(M),zeros(M),zeros(M),zeros(M)
    
    AMPGraph(x, σ, ω, V,
        g, ∂g, A, B,
        params.λ, 
        y, W, W.^2)
end

function init!(amp::AMPGraph, params)
    amp.x .= initx(amp.W, amp.y, params)
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

function initx(A, y, params)
    @extract params: ρ₀ x₀
    N = size(A, 1)  
    if ρ₀ >= 0 && x₀ == :teacher
        x₀ = (1-ρ₀) .* zeros(N) .+ ρ₀ .* params.teacher.x0
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

function oneiter!(amp::AMPGraph, t, params)
    @extract amp: x σ A B ω V g ∂g y W W2 λ
    α = size(W, 2) / size(W, 1)
    ω .= W' * x .-  g ./ α
    
    if t == 1
        β = params.β
        dif_p = mean(y.*( 1/β .* (abs.(ω) .< β))) - 1
        @. g = y * (max(min(ω/β, 1), -1)) - ω
    else
        dif_p = 2/π * sqrt(mean(g.^2)) / (mean(x.^2)) - 1
        @. g = y * sign(ω) - ω
    end
    xnew = W * g .- dif_p .* x

    ## standard equations
    # Maleki's definition of g is not the same of the g in
    # standard equations. Let's call p the maleki's g above.
    # the standard equations read
    
    # @. g = 2/(1+2V) * (y * sign(ω) - ω)
    # @. ∂g = 2/(1+2V) * (2y*δ(ω) - 1)
    # A .= .- W2*∂g
    # B .= W*g .+ A.*x

    # In maleki'algo the regularizer λ is adaptively tuned so that the
    # following holds: 
    # g = p * 2/(1+2V)
    # A =  -div(p)* 2/(1+2V)
    # λ = (1+div(p)) * 2/(1+2V)
    # σ = (1+2V) / 2
    # V = 1/α

    ### here I write the algorithm with fixed lambda,
    ## using the same estimate for div(p) from Jujie's code,
    ## but it doesn't work 
    # V .= W2'*σ 
    # ω .= W' * x .-  V .* g 
    # p = @. g*(1+2V)/2
    # if t == 1
        # β = params.β
        # dif_p = mean(y.*( 1/β .* (abs.(ω) .< β))) - 1
        # @. p = y * (max(min(ω/β, 1), -1)) - ω
    # else 
        ## this is probably wrong at fixed lambda
        # dif_p = 2/π * sqrt(mean(p.^2)) / (mean(x.^2)) - 1
        # @. p = y * sign(ω) - ω
    # end
    # @. g = p*2/(1+2V)
    # @. ∂g = dif_p*2/(1+2V)
    # 
    # A .= .- W2*∂g
    # A .= max.(A, -λ+1e-10)
    # B .= W*g .+ A.*x
    # xnew  = @. B / (A + λ)
    
    Δ = norm(x.-xnew) / length(x)
    x .= xnew 
    Δ
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
            x₀ = :randn, # [:spectral, :teacher,  :randn, nothing, a configuration]
            ρ₀ = 1e-8, # initial overlap with teacher if x₀==:teacher
            verb = 3,
            β = 1, # smoothing coefficient
            ϵ = 1e-10  # stopping criterion
        )

    @extract problem: A y Atst ytst teacher

    seed > 0 && srand(seed)
    N, M = size(A)
    @assert length(y) == M
    # normalize according to  maleki's paper
    scaling = sqrt(1/vecnorm(A)^2 *N)
    y .*= scaling
    A .*= scaling  # E[A_ij^2] = 1/m
    
    params = @NT(β=β,ρ₀=ρ₀, x₀=x₀, λ=λ, teacher=problem.teacher)
    amp = AMPGraph(problem, params)
    
    ## printing utilities ##
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
    ### ####

    epoch = 0
    Δ = 1.
    init!(amp, params)
    report(epoch, Δ, verb+1);# try
    while epoch < epochs
        epoch += 1
        Δ = oneiter!(amp, epoch, params)
        epoch % infotime == 0 && report(epoch, Δ, verb)
        Δ < ϵ && break
    end
    #catch e; e isa InterruptException || error(e); end
    # report(epoch, Δ, verb+1)
    verb > 0 && Δ > ϵ && warn("not converged!")

    return df, amp
end

end #module
