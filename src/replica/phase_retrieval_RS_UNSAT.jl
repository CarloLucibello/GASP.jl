module PhaseRetr

using LittleScienceTools.Roots
using FastGaussQuadrature
using QuadGK
# using AutoGrad
# using Cubature
# import LsqFit: curve_fit
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 15.0
const dx = 0.1

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        # isfinite(r) ? r : 0.0
    end, int..., abstol=1e-6, maxevals=10^7)[1]

# ∫Dexp(f, g=z->1, int=interval) = quadgk(z->begin
#     r = logG(z) + f(z)
#     r = exp(r) * g(z)
# end, int..., abstol=1e-10, maxevals=10^7)[1]

# Numerical Derivaaive
# Can take also directional derivative
# (tune the direction with i and δ).
function deriv(f::Function, i, x...; δ = 1e-6)
    f0 = f(x...)
    ok = false
    while ok == false
        try
            x1 = deepcopy(x) |> collect
            x1[i] .+= δ
            f1 = f(x1...)
            der = (f1-f0) / vecnorm(δ)
            ok = true
        catch
            δ /= 2
        end
    end
    return der
end



let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    global gw
    gw(n::Int) = get!(s, n) do
        (x,w) = gausshermite(n)
        return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
    end
end

function ∫DD(f; n=211)
    (xs, ws) = gw(n)
    s = 0.0
    for (x,w) in zip(xs, ws)
        y = f(x)
        s += w  * ifelse(isfinite(y), y, 0.0)
    end
    return s
end

# deriv(f::Function, i::Integer, x...) = grad(f, i)(x...)


############### PARAMS ################

mutable struct OrderParams
    q0::Float64 # eventually q0=1
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
Gi(q0,δq,qh0,δqh,ρh,ρ) = (q0*δqh - 2*ρ*ρh - δq*qh0)/2

#### ENTROPIC TERM ####

Gs(qh0,δqh,ρh) = 0.5*(Power(ρh,2) + qh0)/δqh

#### ENERGETIC TERM ####

fy(ρ, q0, u0, z0) =  u0 * √(1-ρ^2/q0) + z0 * ρ/√(q0)
fyp(q0, δq, z0, u) = u * √(δq) + z0 * √(q0)

fargGe(y, yp, u) = 1/2 * u^2 + 1/2 * (y^2 - yp^2)^2

function fargGe_min(y, q0, δq, z0; argmin=false)
    ### findmin of 1/2 u^2 + 1/2 * (y^2 - (u √δq + z0 √q0)^2)^2
    a = 9 * √(δq) * √(q0) * z0
    b = 2 * y^2 * δq - 1
    c = 6*(-a + sqrt(complex(a^2 - 6 * b^3)))
    c3 = c^(1/3)
    bc3 = b / c3

    u1 = 1/δq * real((-a/9 -bc3 - c3/6))
    u2 = 1/δq * real((-a/9 + (1+√complex(-3))/2 * bc3 + (1-√complex(-3))/2 * c3/6))
    u3 = 1/δq * real((-a/9 + (1-√complex(-3))/2 * bc3 + (1+√complex(-3))/2 * c3/6))

    roots = [u1,u2,u3]
    if argmin
        m, am_ind = findmin(map(u->fargGe(y, fyp(q0, δq, z0, u), u), roots))
        am = roots[am_ind]
        yp = fyp(q0, δq, z0, am)
        return am, m, yp
    end
    minimum(r->fargGe(y, fyp(q0, δq, z0, r), r), roots)
end

function fargminGe(y, q0, δq, z0)
    ### findmin of 1/2 u^2 + 1/2 * (y^2 - (u √δq + z0 √q0)^2)^2
    a = 9 * √(δq) * √(q0) * z0
    b = 2 * y^2 * δq - 1
    c = 6*(-a + sqrt(complex(a^2 - 6 * b^3)))
    c3 = c^(1/3)
    bc3 = b / c3
    !isfinite(bc3) && (bc3 = zero(Complex)) # guard for b~0  
    u1 = 1/δq * real((-a/9 -bc3 - c3/6))
    u2 = 1/δq * real((-a/9 + (1+√complex(-3))/2 * bc3 + (1-√complex(-3))/2 * c3/6))
    u3 = 1/δq * real((-a/9 + (1-√complex(-3))/2 * bc3 + (1+√complex(-3))/2 * c3/6))

    roots = [u1,u2,u3]
    m, am_ind = findmin(map(u->fargGe(y, fyp(q0, δq, z0, u), u), roots))
    am = roots[am_ind]
    yp = fyp(q0, δq, z0, am)
    return am, m, yp
end


function Ge(q0, δq, ρ, precise; n=81)
    f1 = @spawn Ge(q0, δq, ρ; n=n)
    f2 = @spawn Ge(q0, δq, ρ; n=n+1)
    fetch(f1) / 2 + fetch(f2) / 2
end
function Ge(q0, δq, ρ; n=41)
    -∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            fargGe_min(y, q0, δq, z0; argmin=false)
        end, n=n+150)
    end, n=n)
end


function ∂ρ_Ge_an(q0, δq, ρ, precise; n=41)
    f1 = @spawn ∂ρ_Ge_an(q0, δq, ρ; n=n)
    f2 = @spawn ∂ρ_Ge_an(q0, δq, ρ; n=n+1)
    fetch(f1) / 2 + fetch(f2) / 2
end
function ∂ρ_Ge_an(q0, δq, ρ; n=41)
    -∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            u, f, yp = fargGe_min(y, q0, δq, z0; argmin=true)
            u/√(δq) * (-u0/(√(1-ρ^2/q0))*(ρ/q0) + z0/√(q0)) * y/yp
        end, n=n+150)
    end, n=n)
end

function ∂q0_Ge_an(q0, δq, ρ, precise; n=41)
    f1 = @spawn ∂q0_Ge_an(q0, δq, ρ; n=n)
    f2 = @spawn ∂q0_Ge_an(q0, δq, ρ; n=n+1)
    fetch(f1) / 2 + fetch(f2) / 2
end
function ∂q0_Ge_an(q0, δq, ρ; n=41)
    -∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            u, f, yp = fargGe_min(y, q0, δq, z0; argmin=true)
            u/√(δq) * ((u0/(2*√(1-ρ^2/q0))*(ρ/q0)^2 - z0*ρ/(2*(q0)^(3/2)))*y/yp - (z0/(2*√(q0))))
        end, n=n+150)
    end, n=n)
end

function ∂δq_Ge_an(q0, δq, ρ, precise; n=41)
    f1 = @spawn ∂δq_Ge_an(q0, δq, ρ; n=n)
    f2 = @spawn ∂δq_Ge_an(q0, δq, ρ; n=n+1)
    fetch(f1) / 2 + fetch(f2) / 2
end
function ∂δq_Ge_an(q0, δq, ρ; n=41)
    -∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            u, f, yp = fargGe_min(y, q0, δq, z0; argmin=true)
            -u^2 / (2 * δq)
        end, n=n+150)
    end, n=n)
end

∂ρ_Ge(op, ep) = ∂ρ_Ge_an(op.q0, op.δq, ep.ρ, true; n=81)
∂q0_Ge(op, ep) = ∂q0_Ge_an(op.q0, op.δq, ep.ρ, true; n=81)
∂δq_Ge(op, ep) = ∂δq_Ge_an(op.q0, op.δq, ep.ρ, true; n=81)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 δq qh0 δqh ρh
    @extract ep: α ρ
    Gi(q0,δq,qh0,δqh,ρh,ρ) + Gs(qh0,δqh,ρh) + α*Ge(q0,δq,ρ,true)
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


function stability(op, ep)
    @extract op: q0 δq δqh
    @extract ep: ρ α

    γS = 1/δqh^2 
    # γE = < (ℓ'' / (1+δq*ℓ''))^2 >
    γE = ∫DD(u0->∫DD(z0->begin
            # u, f, yp  = fargGe_min(u0, 1, δq, ρ*u0 + √(q0 - ρ^2)*z0; argmin=true)
            u, f, yp  = fargminGe(u0, 1, δq, ρ*u0 + √(q0 - ρ^2)*z0)
            l2 = 6*yp^2 - 2u0^2 
            (l2 / (1 + δq*l2))^2
        end))

    return 1 - α*γS*γE  # α*γS*γE < 1 => the solution is locally stable
end


#################  SADDLE POINT  ##################

fδqh(op, ep) = -2ep.α * ∂q0_Ge(op, ep)
fqh0(op, ep) = 2ep.α * ∂δq_Ge(op, ep)
fρh(op, ep) = ep.α * ∂ρ_Ge(op, ep)

fq0(op) = (op.ρh^2 + op.qh0) / op.δqh^2
fδq(op) = 1 / op.δqh
fρ(op) = op.ρh / op.δqh

iρh(op, ep) = (true, ep.ρ*op.δqh)

function iδqh(op)
    (true, sqrt(op.qh0 + op.ρh^2))
end

###############################

function fix_inequalities!(op, ep)
    if op.q0 < ep.ρ^2
        op.q0 = sqrt(ep.ρ) + 1e-3
    end
end

function converge!(op::OrderParams, ep::ExtParams, pars::Params;
        fixρ=true, fixnorm=true, extrap=-1)
    @extract pars: maxiters verb ϵ ψ

    fix_inequalities!(op, ep)
    Δ = Inf
    ok = false
    ops = Vector{OrderParams}() # keep some history and extrapolate for quicker convergence
    for it = 1:maxiters
        Δ = 0.0
        ok = oki = true
        verb > 1 && println("it=$it")

        @update  op.qh0    fqh0       Δ ψ verb  op ep
        if fixnorm
            @updateI op.δqh oki   iδqh     Δ ψ verb  op
            ok &= oki
        else
            @update op.δqh  fδqh     Δ ψ verb  op ep
        end
        if fixρ
            @updateI op.ρh oki   iρh   Δ ψ verb  op ep
            ok &= oki
        else
            @update  op.ρh  fρh       Δ ψ verb  op ep
        end

        @update op.δq   fδq       Δ ψ verb  op
        if !fixnorm
            @update op.q0   fq0     Δ ψ verb  op
        end
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op
        end


        verb > 1 && println(" Δ=$Δ\n")
        verb > 2 && it%5==0 && (println(ep);println(all_therm_func(op, ep));println(op))

        @assert isfinite(Δ)
        ok &= Δ < ϵ
        ok && break

        # extrapolation
        extrap > 0 && it > extrap && push!(ops, deepcopy(op))
        if extrap > 0 && it > extrap && it % extrap == 0
            extrapolate!(op, ops)
            empty!(ops)
            verb > 1 && println("# estrapolation -> $op \n")
        end
    end

    ok
end

function converge(;
        q0 = 1.,
        δq=0.5,
        qh0=0., δqh=0.6,
        ρ=0, ρh=0,
        α=0.1,
        ϵ=1e-4, maxiters=100000, verb=3, ψ=0.,
        fixρ=true, fixnorm=true, extrap=-1
    )
    op = OrderParams(q0,δq,qh0,δqh,ρh)
    ep = ExtParams(α, ρ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixρ=fixρ, fixnorm=fixnorm,extrap=extrap)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
        q0=2.0860295957019495,δq=7.816340972218491,qh0=0.034142290508436736,δqh=0.12793710043539372,ρh=0.0012793710043539372,
        ρ=0.01, α=2.,
        ϵ=1e-4, maxiters=10000,verb=3, ψ=0.,
        kws...)

    op = OrderParams(q0,δq,qh0,δqh,ρh)
    ep = ExtParams(first(α), first(ρ))
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; ρ=ρ, α=α, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
    α=1, ρ=1,
    resfile = "results_RS_UNSAT_unconstrained.txt",
    fixρ=true, fixnorm=true, extrap=-1)

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    results = []
    for α in α, ρ in ρ
        fixρ && (ep.ρ = ρ)
        ep.α = α;

        ok = converge!(op, ep, pars; fixρ=fixρ,fixnorm=fixnorm,extrap=extrap)
        tf = all_therm_func(op, ep)
        println("# stability = $(stability(op,ep))")
        push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))
        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
        # !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end

function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    ep = ExtParams(res[line,1:2]...)
    tf = ThermFunc(res[line,3])
    op = OrderParams(res[line,4:end]...)
    return ep, op, tf
end

end ## module
