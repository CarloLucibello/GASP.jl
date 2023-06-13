import GSL: sf_log_erfc
using SpecialFunctions
import CSV
using DataFrames
using Random
using Statistics
using LinearAlgebra
# using Base.Printf: @printf
using ExtractMacro

import ForwardDiff, DiffResults
import ForwardDiff: Dual, value, partials


# using Knet: minibatch
# using Plots; plotlyjs(size=(1000,800))

## Regularizers
abstract type Reg end
struct RegL0 <: Reg end
struct RegL1 <: Reg end
struct RegMCP <: Reg end

Base.print(io::IO, ::RegL0) = print(io, "L0")
Base.print(io::IO, ::RegL1) = print(io, "L1")
Base.print(io::IO, ::RegMCP) = print(io, "MCP")
Reg(r::Symbol) = r == :L0 ? RegL0() : r == :L1 ? RegL1() : RegMCP()
Base.Broadcast.broadcastable(r::Reg) = Ref(r) #opt-out of broadcast

### COMMON TYPES ###
const F = Float64

### SPECIAL FUNCTIONS###

G(x) = exp(-x^2/2) / F(√(2π))
H(x) = erfc(x /F(√2)) / 2
GH(x) = 2 / erfcx(x/F(√2)) / F(√(2π))
HG(x) =  F(√(2π))*erfcx(x/F(√2)) / 2
G(x, Δ) = G(x, 0, Δ)
G(x, μ, Δ) = exp(-(x-μ)^2/(2Δ)) / √(2π*Δ)
logH(x) = sf_log_erfc(x/√2) - log(2)
logG(x, μ, Δ) = -(x-μ)^2/(2Δ) - log(2π*Δ)/2
logG(x) = -x^2/2 - log(2π)/2

lrelu(x, γ=0.1f0) = max(x, γ*x)
log2cosh(x) = abs(x) + log1p(exp(-2abs(x)))
logcosh(x) = log2cosh(x) - log(2)

θfun(x) = x > 0 ? 1 : 0


### AUTOMATIC DIFFERENTIATION ###

grad = ForwardDiff.derivative
@inline function logH(d::Dual{T}) where T
    return Dual{T}(logH(value(d)), -GH(value(d)) * partials(d))
end

# same API as Zygote but using ForwardDiff
gradient(f, x) =  (ForwardDiff.gradient(f, x),)

# same API as Zygote but using ForwardDiff
function withgradient(f, x)
    out = DiffResults.GradientResult(x)
    ForwardDiff.gradient!(out, f, x)
    return DiffResults.value(out),  (DiffResults.gradient(out),)
end

### PROBLEMS ###

mutable struct Problem
    A::Matrix       # measurement matrix of size N x M 
    y::Vector       # measured vector of size M
    Atst::Matrix
    ytst::Vector
    teacher         # informations on the teacher
    student         # informations on the student
    seed::Int
    name::String
end

Problem(xtrn ,ytrn ,xtst, ytst) = Problem(xtrn ,ytrn ,xtst, ytst, nothing, -1, "")

Problem(name::String; kws...) = name == "gle" ? problem_gle(; kws...) :
                        error("uknown problem")

function Base.show(io::IO, prob::Problem)
    print(io, "Problem \"$(prob.name)\":\n")
    for f in fieldnames(Problem)
        print(io, "  $f: $(summary(getfield(prob, f)))\n")
    end
end

# Generilized Linear Estimation
function problem_gle(; N=400, α=0.4, seed=-1,
            TS=true,      # teacher-student problem
            act = abs,
            Δ0=0., Δ=Δ0,  #teacher and students noise variance
            prior0=:gauss,
            varx0=1,  # variance of the gaussian part  of x0
            prior=prior0,
            ρ=1. # fraction non-zeros for :gauss prior
        )

    @assert TS # only teacher student problems for the time being
    seed > 0 && Random.seed!(seed)
    M = round(Int, N*α)
    Mtst = round(Int, max(200, M÷4))
    if prior0 == :gauss
        x0 = (sqrt(varx0) .*randn(N)) .* (rand(N) .< ρ)
    elseif prior0 == :gausspos
        x0 = sqrt(varx0) .* abs.(randn(N))
    end
    # project_sphere!(x0)
    A = randn(F, N, M) ./ √N
    y = act.(A'x0) .+ √Δ0 .* randn(M)

    Atst = randn(F, N, Mtst) ./ √N
    ytst = act.(Atst'x0) .+ √Δ0 .* randn(Mtst)

    teacher = (prior0=prior0, x0=x0, Δ0=Δ0, act=act, α=α, ρ=ρ, varx0=varx0)
    student = (prior=prior, Δ=Δ)
    return Problem(A, y, Atst, ytst, teacher, student, seed, "gle $act")
end

###### PHASE RETRIEVAL Specific #############

# optimal spectral init (Mondelli, Montanari '17)
# real case
function spectral_init_optimal(A::Matrix, y::Vector)
    N, M = size(A)
    @assert length(y) == M
    α = M / N
    yp = max.(0, y)
    T = Diagonal((yp .- 1) ./ (yp .+ √(2α) .- 1))
    D = Symmetric(1/M * A*T*A')
    λ, v = eigen(D)
    _, imax = findmax(λ)
    vmax = v[:,imax]
    project_sphere!(vmax)
    vmax
end


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
    # initial condition :randn
    # gives q_0 = 1 + ρ₀^2
    #       ρ   = ρ₀
    x .= x .+ ρ₀ .* problem.teacher.x0
    x
end

### Mathematica Compatibility ####

Power(x,y) = x^y
Log(x) = log(x)

########## UTILITY FUNCTIONS #############
function batches(x, y, batchsize; kws...)
    @assert size(x, 2) == length(y)
    size(x, 2) == batchsize && return [(x, y)]
    minibatch(x, y, batchsize; kws...)
end

percent(x) = x*100

overlap(W, Wt) = dot(W, Wt) / (norm(Wt) * norm(W))

norm2(x::Vector) = norm(x)^2
MSE(ŷ, y) = (d = y .- ŷ; sum(d.*d)/ length(d))

vecnorm(x) = norm(x)

# project on the hypersphere
function project_sphere!(σ::Vector)
    σ .*= √length(σ) / norm(σ)
    σ
end

# remove from g the component in
# the direction of σ
function remove_projection!(g, σ)
    g .-= σ .* (dot(g, σ)/dot(σ, σ))
end

nobs(x) = size(x, ndims(x))

macro update(x, func, Δ, ψ, verb, params...)
#     nx = x isa Symbol ? x :
#          x.head == :. ? x.args[2] : x
#     dump(nx)
    # dump(x)
    # name = string(x)
    name = string(x.args[2].value)

    # if x isa Symbol || x.head == :ref
        # name = string(x.args[1], " ", eval(x.args[2]))
    # else
    #     name = string(x.args[2].args[1])
    # end
    x = esc(x)
    Δ = esc(Δ)
    ψ = esc(ψ)
    verb = esc(verb)
    params = map(esc, params)
    fcall = Expr(:call, func, params...)
    quote
        oldx = $x
        newx = $fcall
        abserr = norm(newx - oldx)
        relerr = abserr == 0 ? 0 : abserr / ((norm(newx) + norm(oldx)) / 2)
        $Δ = max($Δ, min(abserr, relerr))
        $x = (1 - $ψ) * newx + $ψ * oldx
        $verb > 1 && println("  ", $name, " = ", $x)
    end
end

macro updateI(x, ok, func, Δ, ψ, verb, params...)
    name = string(x.args[2].value)
    x = esc(x)
    ok = esc(ok)
    Δ = esc(Δ)
    ψ = esc(ψ)
    verb = esc(verb)
    func = esc(func)
    params = map(esc, params)
    fcall = Expr(:call, func, params...)
    quote
        oldx = $x
        $ok, newx = $fcall
        if $ok
            abserr = abs(newx - oldx)
            relerr = abserr == 0 ? 0 : abserr / ((abs(newx) + abs(oldx)) / 2)
            $Δ = max($Δ, min(abserr, relerr))
            $x = (1 - $ψ) * newx + $ψ * oldx
            $verb > 1 && println("  ", $name, " = ", $x)
        else
            $verb > 1 && println("  ", $name, " = ", $x)
        end
    end
end


######## FILE PRINTING ######################

# compact printing
cprintln(io::IO, x) = (cprint(io, x); println(io))
cprintln(x) = cprintln(stdout, x)
cprint(x) = cprint(stdout, x)

cprint(io::IO, x) = print(io, "$(round4(x)) ")

round4(x) = x
round4(x::Number) = round(x, digits=4)
round4(x::NamedTuple) = map(round4, x)

# function cprint(io::IO, x::Pair{S, Float64}) where {S}
    # if 1e-3 < x[2] < 1e3
        # @printf(io, "%s=%.5f", x[1], x[2])
    # else
    # @printf(io, "%s=%.3e", x[1], x[2])
    # end
# end

writedf(file, df::DataFrame) = CSV.write(file, df, delim='\t')

function readdf(file; types = Dict())
    df = CSV.read(file, delim='\t', allowmissing=:none)
    for (c, T) in types
        if T <: Array
            df[c] = eval.(parse.(df[c]))
        end
    end
    df
end

function exclusive(f::Function, fn::AbstractString = "lock.tmp")
    run(`lockfile -1 $fn`)
    try
        f()
    finally
        run(`rm -f $fn`)
    end
end


struct NewtonParameters
    δ::Float64
    ϵ::Float64
    verb::Int
    maxiters::Int
end

δ = 1e-8

function ∇!(∂f::Matrix, f::Function, x0, δ, f0, x1)
    n = length(x0)
    copy!(x1, x0)
    for i = 1:n
        x1[i] += δ
        @. ∂f[:,i] = (f(x1) - f0) / δ
        x1[i] = x0[i]
    end
end

function newton(f::Function, x₀, pars::NewtonParameters)
    η = 1.0
    n = length(x₀)
    ∂f = Array{Float64}(undef, n, n)
    x = Float64[x₀[i] for i = 1:n]
    x1 = Array{Float64}(undef, n)

    f0 = f.(x)
    @assert length(f0) == n
    @assert isa(f0, Union{Real,Vector})
    normf0 = norm(f0)
    it = 0
    while normf0 ≥ pars.ϵ
        it > pars.maxiters && return (false, x, it, normf0)
        it += 1
        if pars.verb > 1
            println("it=$it")
            println("  x=$x")
            println("  f0=$f0")
            println("  norm=$(vecnorm(f0))")
            println("  η=$η")
        end
        δ = pars.δ
        while true
            try
                ∇!(∂f, f, x, δ, f0, x1)
                break
            catch
                δ /= 2
            end
            δ < 1e-15 && return (false, x, it, normf0)
        end
        if typeof(f0) == Vector
            Δx = -∂f \ f0
        else
            Δx = -f0 / ∂f[1,1]
        end
        pars.verb > 1 && println("  Δx=$Δx")
        while true
            for i = 1:n
                x1[i] = x[i] + Δx[i] * η
            end
            local new_f0, new_normf0
            try
                new_f0 = f.(x1)
                new_normf0 = norm(new_f0)
            catch
                new_normf0 = Inf
            end
            if new_normf0 < normf0
                η = min(1.0, η * 1.1)
                if isa(f0, Vector)
                    copy!(f0, new_f0)
                else
                    f0 = new_f0
                end
                normf0 = new_normf0
                copy!(x, x1)
                break
            end
            η /= 2
            η < 1e-15 && return (false, x, it, normf0)
        end
    end
    return true, x, it, normf0
end

function shortshow(io::IO, x)
    T = typeof(x)
    print(io, T.name.name, "(", join([string(f, "=", getfield(x, f)) for f in fieldnames(T)], ","), ")")
end

function plainshow(x)
    T = typeof(x)
    join([getfield(x, f) for f in fieldnames(T)], " ")
end

function headershow(io::IO, T::Type, i0 = 0)
    print(io, join([string(i+i0,"=",f) for (i,f) in enumerate(fieldnames(T))], " "))
    return i0 + length(fieldnames(T))
end

function headershow(io::IO, x::String, i0 = 0)
    i0 += 1
    print(io, string(i0,"=",x," "))
    i0
end

function allheadersshow(io::IO, x...)
    i0 = 0
    print(io, "#")
    for y in x
        i0 = headershow(io, y, i0)
        print(io, " ")
    end
    println(io)
end
