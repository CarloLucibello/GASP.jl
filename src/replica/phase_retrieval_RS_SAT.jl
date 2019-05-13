module PhaseRetr

using LittleScienceTools.Roots
using QuadGK
using AutoGrad
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 15.0
const dx = 0.02

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-7, maxevals=10^7)[1]

# ∫Dexp(f, g=z->1, int=interval) = quadgk(z->begin
#     r = logG(z) + f(z)
#     r = exp(r) * g(z)
# end, int..., abstol=1e-10, maxevals=10^7)[1]

# Numerical Derivaaive
# Can take also directional derivative
# (tune the direction with i and δ).
function deriv(f::Function, i, x...; δ = 1e-7)
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
    qh1::Float64
    ρh::Float64
end

mutable struct ExtParams
    α::Float64
    ρ::Float64
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
Gi(q0,qh0,qh1,ρh,ρ) = (-qh1 - 2*ρ*ρh + q0*qh0)/2

#### ENTROPIC TERM ####

Gs(qh0,qh1,ρh) = 0.5*(ρh^2 + qh0)/(qh0 - qh1) -0.5*log(qh0-qh1)

#### ENERGETIC TERM ####

fy(q0, ρ, u0, z0) = (√(1 - ρ^2/q0) * u0 + ρ/√q0 * z0)^2

function argGe(y, q0, z0)
    -(q0*z0^2+y)/(2(1-q0)) -0.5*log(2π*y*(1-q0)) + log(cosh(√(y*q0)*z0/(1-q0)))
end

function Ge(q0, ρ)
    res= ∫D(z0->begin
        ∫D(u0->begin
            y = fy(q0, ρ, u0, z0)
            argGe(y, q0, z0)
        end)
    end)
    β=3.; p=3
    # 0.1*(0.5*β^2/2*(1 - q0^p) + β*ρ^p) + 0.9*res
    res
end

∂q0_Ge(q0, ρ) = deriv(Ge, 1, q0, ρ)
∂ρ_Ge(q0, ρ) = deriv(Ge, 2, q0, ρ)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 qh0 qh1 ρh
    @extract ep: α ρ
    Gi(q0,qh0,qh1,ρh,ρ) + Gs(qh0,qh1,ρh) + α*Ge(q0,ρ)
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
fqh0(q0, ρ, α) = -2α * ∂q0_Ge(q0, ρ)
fρh(q0, ρ, α) = α * ∂ρ_Ge(q0, ρ)

fq0(qh0,qh1,ρh) = (qh0 + Power(ρh,2))/Power(qh0 - qh1,2)
fρ(qh0,qh1,ρh) = ρh/(qh0-qh1)
fq1(qh0,qh1,ρh) = (2*qh0 - qh1 + Power(ρh,2))/(Power(qh0 - qh1,2))

iρh(ρ,qh0,qh1) = (true, ρ*(qh0-qh1))

function iqh1(qh0, qh1₀, ρh)
    ok, qh1, it, normf0 = findroot(qh1 -> fq1(qh0,qh1,ρh) - 1, qh1₀, NewtonMethod(atol=1e-8))
    ok || error("iqh1 failed: iδqh=$(qh1), it=$it, normf0=$normf0")
    return ok, qh1
end

###############################


function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixρ = true)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    it = 0
    for it = 1:maxiters
        Δ = 0.0
        verb > 1 && println("it=$it")

        @update  op.qh0    fqh0       Δ ψ verb  op.q0 ep.ρ ep.α
        @updateI op.qh1 ok   iqh1     Δ ψ verb  op.qh0 op.qh1 op.ρh
        if fixρ
            @updateI op.ρh ok   iρh   Δ ψ verb  ep.ρ op.qh0 op.qh1
        else
            @update  op.ρh  fρh       Δ ψ verb  op.q0 ep.ρ ep.α
        end
        # op.qh0 < 0 &&( op.qh0=0) 
        # fix_inequalities_hat!(op, ep)
        # fix_inequalities_nonhat!(op, ep)

        @update op.q0   fq0       Δ ψ verb  op.qh0 op.qh1 op.ρh
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op.qh0 op.qh1 op.ρh
        end

        # op.q0 < 0 &&( op.q0=0.1) 
        
        verb > 1 && println(" Δ=$Δ\n")
        verb > 1 && println(all_therm_func(op, ep))

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end

    ok
end

function converge(;
        q0=0.5,
        qh0=0., qh1=0.6,
        ρ=0, ρh=0,
        α=0.1,
        ϵ=1e-5, maxiters=100000, verb=2, ψ=0.,
        fixm = false, fixρ=true
    )
    op = OrderParams(q0,qh0,qh1,ρh)
    ep = ExtParams(α, ρ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixρ=fixρ)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
    q0=0.3188,
    qh0=0.36889,qh1=0.36889, ρh=0.56421,
    ρ=0.384312, α=1,
    ϵ=1e-5, maxiters=10000,verb=2, ψ=0.,
    kws...)

    op = OrderParams(q0,qh0,qh1,ρh)
    ep = ExtParams(first(α), first(ρ))
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; ρ=ρ,α=α, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
    α=1, ρ=1,
    resfile = "results.txt",
    fixm=false, fixρ=false, mlessthan1=false)

    if !isfile(resfile)
        open(resfile, "w") do f
            allheadersshow(f, ExtParams, ThermFunc, OrderParams)
        end
    end

    results = []
    for α in α, ρ in ρ
        fixρ && (ep.ρ = ρ)
        ep.α = α;

        ok = converge!(op, ep, pars; fixρ=fixρ)
        
        tf = all_therm_func(op, ep)
        push!(results, (ok, deepcopy(ep), deepcopy(tf), deepcopy(op)))
        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end


function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    ep = ExtParams(res[line,1:2]...)
    op = OrderParams(res[line,3:6]...)
    tf = ThermFunc(res[line,end])
    return ep, op, tf
end

end ## module
