module Perc

using LittleScienceTools.Roots
using QuadGK
using LsqFit
include("../../common.jl")


###### INTEGRATION  ######
const ∞ = 30.0
const dx = 0.002

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-8, maxevals=10^7)[1]

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


############### PARAMS ################

mutable struct OrderParams
    δq::Float64
    qh0::Float64
    δqh::Float64
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
Gi(δq,qh0,δqh,ρh,ρ) = (δqh - 2*ρ*ρh - δq*qh0)/2

#### ENTROPIC TERM ####

Gs(qh0,δqh,ρh) = 0.5*(Power(ρh,2) + qh0)/δqh

#### ENERGETIC TERM ####

function argGe(δq,z0)
    c =  z0 / √δq
    c > 0 && return 0.
    1/2 * min(c^2, 4.)    
end

function ∂δq_argGe(δq,z0)
    c =  z0 / √δq
    c > 0 && return 0.
    c < -2 && return 0
    -0.5*z0^2 / (δq)^2
end

function Ge(δq, ρ)
    -2∫D(z0->begin
        l = argGe(δq, z0)
        l * H(-√(ρ/(1-ρ^2))*z0)  # teacher-student
        # l/2   # student
    end)
end

function ∂δq_Ge(δq, ρ)
    -2∫D(z0->begin
        l = ∂δq_argGe(δq, z0)
        l * H(-√(ρ^2/(1-ρ^2))*z0)  # teacher-student
        # l/2   # student
    end)
end

# function ∂δq_Ge(δq, ρ)
#     -2∫Dexp(z0 -> logH(-√(ρ^2/(1-ρ^2))*z0),
#             z0 -> ∂δq_argGe(δq, z0))
# end

function ∂ρ_Ge(δq, ρ)
    -2∫D(z0->begin
        l = argGe(δq, z0)
        dH = G(-√(ρ^2/(1-ρ^2))*z0) * (z0/Power(1 - Power(ρ,2),1.5))
        l * dH  # teacher-student
    end)
end

# function ∂ρ_Ge(δq, ρ)
#     -2∫D(z0 -> logG(-√(ρ^2/(1-ρ^2))*z0),
#         z0 -> argGe(δq, z0)*(z0/Power(1 - Power(ρ,2),1.5)))
# end

# ∂δq_Ge(δq, ρ) = deriv(Ge, 1, δq, ρ)  
# ∂ρ_Ge(δq, ρ) = deriv(Ge, 2, δq, ρ)  

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: δq qh0 δqh ρh 
    @extract ep: α ρ
    Gi(δq,qh0,δqh,ρh,ρ) + Gs(qh0,δqh,ρh) + α*Ge(δq,ρ)
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
fqh0(δq, ρ, α) = 2α * ∂δq_Ge(δq, ρ)
fρh(δq, ρ, α) = α * ∂ρ_Ge(δq, ρ)

fδq(qh0,δqh,ρh) = 1/δqh
fρ(qh0,δqh,ρh) = ρh/δqh

iρh(ρ,qh0,δqh) = (true, ρ*δqh)

function iδqh(qh0, δqh, ρh)
    (true, sqrt(qh0 + ρh^2))
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

        @update  op.qh0    fqh0       Δ ψ verb  op.δq ep.ρ ep.α
        @updateI op.δqh ok   iδqh     Δ ψ verb  op.qh0 op.δqh op.ρh
        if fixρ
            @updateI op.ρh ok   iρh   Δ ψ verb  ep.ρ op.qh0 op.δqh
        else
            @update  op.ρh  fρh       Δ ψ verb  op.δq ep.ρ ep.α
        end

        # fix_inequalities_hat!(op, ep)
        # fix_inequalities_nonhat!(op, ep)        

        @update op.δq   fδq       Δ ψ verb  op.qh0 op.δqh op.ρh
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op.qh0 op.δqh op.ρh
        end


        verb > 1 && println(" Δ=$Δ\n")
        verb > 1 && println(all_therm_func(op, ep))

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end

    ok
end

function converge(;
        δq=0.5,
        qh0=0., δqh=0.6,
        ρ=0, ρh=0,
        α=0.1,
        ϵ=1e-6, maxiters=100000, verb=2, ψ=0.,
        fixm = false, fixρ=true
    )
    op = OrderParams(δq,qh0,δqh,ρh)
    ep = ExtParams(α, ρ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixρ=fixρ)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
    δq=0.3188,
    qh0=0.36889,δqh=0.36889, ρh=0.56421,
    ρ=0.384312, α=1,
    ϵ=1e-6, maxiters=10000,verb=2, ψ=0.,
    kws...)

    op = OrderParams(δq,qh0,δqh,ρh)
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
            allheadersshow(f, ExtParams, OrderParams, ThermFunc)
        end
    end

    results = []
    for α in α, ρ in ρ
        fixρ && (ep.ρ = ρ)
        ep.α = α; 

        ok = converge!(op, ep, pars; fixρ=fixρ)
        tf = all_therm_func(op, ep)
        push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))
        if ok
            open(resfile, "a") do rf
                println(rf, plainshow(ep), " ", plainshow(op), " ", plainshow(tf))
            end
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
