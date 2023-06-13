module PhaseRetr

using LittleScienceTools.Roots
using QuadGK
using AutoGrad
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 20.0
const dx = 0.01

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-10, maxevals=10^7)[1]

# ∫Dexp(f, g=z->1, int=interval) = quadgk(z->begin
#     r = logG(z) + f(z)
#     r = exp(r) * g(z)
# end, int..., abstol=1e-10, maxevals=10^7)[1]

# Numerical Derivative
# Can take also directional derivative
# (tune the direction with i and δ).
function deriv(f::Function, i, x...; δ = 1e-6)
    x1 = deepcopy(x) |> collect
    x1[i] .+= δ
    f0 = f(x...)
    f1 = f(x1...)
    return (f1-f0) / vecnorm(δ)
end

# deriv(f::Function, i::Integer, x...) = grad(f, i)(x...)


############### PARAMS ################

mutable struct OrderParams
    q0::Float64
    qh0::Float64
end

mutable struct ExtParams
    α::Float64
end

mutable struct Params
    ϵ::Float64 # stop criterium
    ψ::Float64 # dumping
    maxiters::Int
    verb::Int
end

mutable struct ThermFunc
    ϕ::Float64
end

Base.show(io::IO, op::OrderParams) = shortshow(io, op)
Base.show(io::IO, ep::ExtParams) = shortshow(io, ep)
Base.show(io::IO, params::Params) = shortshow(io, params)
Base.show(io::IO, tf::ThermFunc) = shortshow(io, tf)

###################################################################################

# Mathematica compatibility
Power(x,y) = x^y
Log(x) = log(x)

#### INTERACTION TERM ####
Gi(q0,qh0) = (-q0*qh0)/2

#### ENTROPIC TERM ####

Gs(qh0) = 0.5*(qh0^2 + qh0)/(qh0 + 1) -0.5*log(qh0 + 1)

#### ENERGETIC TERM ####

fy(q0, u0, z0) = (√(1 - q0) * u0 + √q0 * z0)^2

function argGe(y, q0, z0)
    -(q0*z0^2+y)/(2(1-q0)) -0.5*log(2π*y*(1-q0)) + logcosh(√(y*q0)*z0/(1-q0))
end

function Ge(q0)
    res=∫D(z0->begin
        ∫D(u0->begin
            y = fy(q0, u0, z0)
            argGe(y, q0, z0)
        end)
    end)
    res
end

∂q0_Ge(q0) = deriv(Ge, 1, q0)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 qh0
    @extract ep: α
    Gi(q0,qh0) + Gs(qh0) + α*Ge(q0)
end

## Thermodinamic functions
# The energy of the pure states selected by x
# E = -∂(m*ϕ)/∂m
# if working at fixed x. If x is optimized Σ=0 and
# E = -ϕ
# so this formula is valid both at fixed and at optimized x
function all_therm_func(op::OrderParams, ep::ExtParams)
    ϕ = free_entropy(op, ep)
    return ThermFunc(ϕ)
end

#################  SADDLE POINT  ##################
fqh0(q0, α) = 2α * ∂q0_Ge(q0)

fq0(qh0) = qh0/(1+qh0)

iqh0(q0, qh0₀) = (true, q0/(1-q0))
###############################


function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixq0 = false)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false
    it = 0
    for it = 1:maxiters
        Δ = 0.0
        verb > 1 && println("it=$it")
        if fixq0 
            @updateI op.qh0 ok  iqh0       Δ ψ verb  op.q0 op.qh0
        else 
            @update  op.qh0    fqh0       Δ ψ verb  op.q0 ep.α
            @update op.q0      fq0       Δ ψ verb  op.qh0
        end

        verb > 1 && println(" Δ=$Δ\n")
        verb > 2 && println(all_therm_func(op, ep))

        @assert isfinite(Δ)
        ok = Δ < ϵ
        ok && break
    end

    ok
end

function converge(;
        q0=0.5,
        qh0=0.,
        α=0.1,
        ϵ=1e-5, maxiters=100000, verb=2, ψ=0.,
        fixq0 = false
    )
    op = OrderParams(q0,qh0)
    ep = ExtParams(α)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixq0=fixq0)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
    q0=0.3188,
    qh0=0.36889,
    α=1,
    ϵ=1e-5, maxiters=10000,verb=2, ψ=0.,
    kws...)

    op = OrderParams(first(q0),qh0)
    ep = ExtParams(first(α))
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; q0=q0, α=α, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
        α=1, q0=0.3,
        resfile = "results.txt",
        fixq0=false)

    if !isfile(resfile)
        open(resfile, "w") do f
            allheadersshow(f, ExtParams, ThermFunc, OrderParams)
        end
    end

    results = []
    for α in α, q0 in q0
        fixq0 && (op.q0 = q0)
        ep.α = α
        pars.verb > 0 && println("# α=$α  q0=$q0")
        ok = converge!(op, ep, pars; fixq0=fixq0)
        
        tf = all_therm_func(op, ep)
        push!(results, (ok, deepcopy(ep), deepcopy(tf), deepcopy(op)))
        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
        !ok && break
        pars.verb > 0 && print(tf,"\n")
    end
    return results
end


# function readparams(file, line = 0)
#     res = readdlm(file)
#     line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
#     ep = ExtParams(res[line,1:2]...)
#     op = OrderParams(res[line,3:6]...)
#     tf = ThermFunc(res[line,end])
#     return ep, op, tf
# end

end ## module