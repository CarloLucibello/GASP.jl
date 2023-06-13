module PhaseRetr

# using LittleScienceTools.Roots
using FastGaussQuadrature
using QuadGK
using AutoGrad
using Cubature
using Cuba
using IterTools: product

# import LsqFit: curve_fit
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 20.0
const dx = 0.05

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-7, maxevals=10^7)[1]

## Cubature.jl
∫∫∫D(f, xmin::Vector, xmax::Vector) = hcubature(z->begin
    r = G(z[1])*G(z[2])*G(z[3])*f(z[1],z[2],z[3])
    isfinite(r) ? r : 0.0
end, xmin, xmax, abstol=1e-7)[1]


function ∫∫∫D(f)
    ints = [(interval[i],interval[i+1]) for i=1:length(interval)-1]
    intprods = product(ints, ints, ints)
    # @show collect(intprods)
    sum(ip-> begin
            xmin = [ip[1][1],ip[2][1],ip[3][1]]
            xmax = [ip[1][2],ip[2][2],ip[3][2]]
            ∫∫∫D(f, xmin, xmax)
        end, intprods)
end

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

function ∫DD(f; n=161)
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
    Δ::Float64
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

fy(ρ, q0, u0, z0) =  z0
fyp(q0, δq, z0, u) = u * √(δq) + z0

fargGe(y, yp, u) = 1/2 * u^2 + 1/2 * (y - yp^2)^2

function fargGe_min(y, q0, δq, z0; argmin=false)
    ### findmin of 1/2 u^2 + 1/2 * (y - u^2 δq - 2 u z0 √δq)^2
    a = 9 * √(δq) * √(q0) * z0
    b = 2 * y * δq - 1
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


function Ge(q0, δq, ρ, Δ, precise; n=41)
    f1 = @spawn Ge(q0, δq, ρ, Δ; n=n)
    f2 = @spawn Ge(q0, δq, ρ, Δ; n=n+1)
    fetch(f1) / 2 + fetch(f2) / 2
end
function Ge(q0, δq, ρ, Δ; n=41)
    # -∫DD(u0->begin
        -∫DD(z0->begin
            ys = fy(ρ, q0, 0., z0)
            ∫DD(y->begin
                fargGe_min(√Δ*y + ys^2, q0, δq, z0; argmin=false)
            end, n=100)
        end, n=n+160)
    # end, n=n)
    # -∫∫∫D((u0,z0,y)->begin
    #         ys = fy(ρ, q0, u0, z0)
    #         fargGe_min(√Δ*y + ys^2, q0, δq, z0; argmin=false)
    #     end)
end


function ∂ρ_Ge_an(q0, δq, ρ, Δ, precise; n=41)
    f1 = @spawn ∂ρ_Ge_an(q0, δq, ρ, Δ; n=n)
    f2 = @spawn ∂ρ_Ge_an(q0, δq, ρ, Δ; n=n+1)
    fetch(f1) / 2 + fetch(f2) / 2
end
function ∂ρ_Ge_an(q0, δq, ρ, Δ; n=41)
    # -∫DD(u0->begin
        -∫DD(z0->begin
            ys = fy(ρ, q0, 0., z0)
            ∫DD(y->begin
                u, f, yp = fargGe_min(√Δ*y + ys^2, q0, δq, z0; argmin=true)
                (√Δ*y + ys^2 - yp^2) * (-u0/(√(1-ρ^2/q0))*(ρ/q0) + z0/√(q0)) * 2ys
            end, n=100)
        end, n=n+160)
    # end, n=n)
    # -∫∫∫D((u0,z0,y)->begin
    #         ys = fy(ρ, q0, u0, z0)
    #         u, f, yp = fargGe_min(√Δ*y + ys^2, q0, δq, z0; argmin=true)
    #         (√Δ*y + ys^2 - yp^2) * (-u0/(√(1-ρ^2/q0))*(ρ/q0) + z0/√(q0)) * 2ys
    #     end)
end

function ∂q0_Ge_an(q0, δq, ρ, Δ, precise; n=41)
    f1 = @spawn ∂q0_Ge_an(q0, δq, ρ, Δ; n=n)
    f2 = @spawn ∂q0_Ge_an(q0, δq, ρ, Δ; n=n+1)
    fetch(f1) / 2 + fetch(f2) / 2
end
function ∂q0_Ge_an(q0, δq, ρ, Δ; n=41)
    # -∫DD(u0->begin
        -∫DD(z0->begin
            ys = fy(ρ, q0, 0., z0)
            ∫DD(y->begin
                u, f, yp = fargGe_min(√Δ*y + ys^2, q0, δq, z0; argmin=true)
                (√Δ*y + ys^2 - yp^2) * ((u0/(2*√(1-ρ^2/q0))*(ρ/q0)^2 - z0*ρ/(2*(q0)^(3/2)))*2ys -z0/(2*√(q0))*2yp)
            end, n=100)
        end, n=n+160)
    # end, n=n)
    # -∫∫∫D((u0,z0,y)->begin
    #         ys = fy(ρ, q0, u0, z0)
    #         u, f, yp = fargGe_min(√Δ*y + ys^2, q0, δq, z0; argmin=true)
    #         (√Δ*y + ys^2 - yp^2) * ((u0/(2*√(1-ρ^2/q0))*(ρ/q0)^2 - z0*ρ/(2*(q0)^(3/2)))*2ys -z0/(2*√(q0))*2yp)
    #     end)
end

function ∂δq_Ge_an(q0, δq, ρ, Δ, precise; n=41)
    f1 = @spawn ∂δq_Ge_an(q0, δq, ρ, Δ; n=n)
    f2 = @spawn ∂δq_Ge_an(q0, δq, ρ, Δ; n=n+1)
    fetch(f1) / 2 + fetch(f2) / 2
end
function ∂δq_Ge_an(q0, δq, ρ, Δ; n=41)
    # -∫DD(u0->begin
        -∫DD(z0->begin
            ys = fy(ρ, q0, 0., z0)
            ∫DD(y->begin
                u, f, yp = fargGe_min(√Δ * y + ys^2, q0, δq, z0; argmin=true)
                (-u^2 / (2 * δq))
            end, n=100)
        end, n=n+160)
    # end, n=n)
    # -∫∫∫D((u0,z0,y)->begin
    #         ys = fy(ρ, q0, u0, z0)
    #         u, f, yp = fargGe_min(√Δ*y + ys^2, q0, δq, z0; argmin=true)
    #         (-u^2 / (2 * δq))
    #     end)
end

∂ρ_Ge(op, ep) = ∂ρ_Ge_an(op.q0, op.δq, ep.ρ, ep.Δ, true; n=41)
∂q0_Ge(op, ep) = ∂q0_Ge_an(op.q0, op.δq, ep.ρ, ep.Δ, true; n=41)
∂δq_Ge(op, ep) = ∂δq_Ge_an(op.q0, op.δq, ep.ρ, ep.Δ, true; n=41)

# ∂ρ_Ge(op, ep) = ∂ρ_Ge_an(op.q0, op.δq, ep.ρ, ep.Δ)
# ∂q0_Ge(op, ep) = ∂q0_Ge_an(op.q0, op.δq, ep.ρ, ep.Δ)
# ∂δq_Ge(op, ep) = ∂δq_Ge_an(op.q0, op.δq, ep.ρ, ep.Δ)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 δq qh0 δqh ρh
    @extract ep: α ρ Δ
    Gi(q0,δq,qh0,δqh,ρh,ρ) + Gs(qh0,δqh,ρh) + α*Ge(q0,δq,ρ,Δ,true)
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

fδqh(op, ep) = 1/op.δq #-2ep.α * ∂q0_Ge(op, ep)
fqh0(op, ep) = 2ep.α * ∂δq_Ge(op, ep)
# fqh0(op, ep) = 2ep.α * (Ge(1., op.δq+1e-8, 1., ep.Δ)-Ge(1., op.δq-1e-8, 1., ep.Δ))/2e-8
fρh(op, ep) = 1/op.δq#ep.α * ∂ρ_Ge(op, ep)

fq0(op) = (op.ρh^2 + op.qh0) / op.δqh^2
fδq(op) = 1 / op.δqh
fρ(op) = op.ρh / op.δqh

iρh(op, ep) = (true, ep.ρ*op.δqh)

function iδqh(op)
    (true, sqrt(op.qh0 + op.ρh^2))
end

# fqh0(δq, ρ, α) = 2α * ∂δq_Ge(δq, ρ)
# fρh(δq, ρ, α) = α * ∂ρ_Ge(δq, ρ)
#
# fδq(qh0,δqh,ρh) = 1/δqh
# fρ(qh0,δqh,ρh) = ρh/δqh
#
# iρh(ρ,qh0,δqh) = (true, ρ*δqh)
#
# function iδqh(qh0, δqh, ρh)
#     (true, sqrt(qh0 + ρh^2))
# end

###############################

function fix_inequalities!(op, ep)
    if op.q0 < ep.ρ^2
        op.q0 = sqrt(ep.ρ) + 1e-3
    end
end

function converge!(op::OrderParams, ep::ExtParams, pars::Params;
        fixρ=true, fixnorm=true, extrap=-1)
    @extract pars: maxiters verb ϵ ψ

    fixnorm && (op.q0 = 1)
    fix_inequalities!(op, ep)
    Δ = Inf
    ok = false
    ops = Vector{OrderParams}() # keep some history and extrapolate for quicker convergence
    for it = 1:maxiters
        Δ = 0.0
        ok = oki = true
        verb > 1 && println("it=$it")

        @update  op.qh0    fqh0       Δ ψ verb  op ep
        if fixρ
            @updateI op.ρh oki   iρh   Δ ψ verb  op ep
            ok &= oki
        else
            @update  op.ρh  fρh       Δ ψ verb  op ep
        end
        if fixnorm
            @updateI op.δqh oki   iδqh     Δ ψ verb  op
            ok &= oki
        else
            @update op.δqh  fδqh     Δ ψ verb  op ep
        end

        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op
        end
        @update op.δq   fδq       Δ 0.99 verb  op
        if !fixnorm
            @update op.q0   fq0     Δ ψ verb  op
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

# function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixρ = true)
#     @extract pars : maxiters verb ϵ ψ
#
#     Δ = Inf
#     ok = false
#
#     it = 0
#     for it = 1:maxiters
#         Δ = 0.0
#         verb > 1 && println("it=$it")
#
#         @update  op.qh0    fqh0       Δ ψ verb  op.δq ep.ρ ep.α
#         @updateI op.δqh ok   iδqh     Δ ψ verb  op.qh0 op.δqh op.ρh
#         if fixρ
#             @updateI op.ρh ok   iρh   Δ ψ verb  ep.ρ op.qh0 op.δqh
#         else
#             @update  op.ρh  fρh       Δ ψ verb  op.δq ep.ρ ep.α
#         end
#
#         # fix_inequalities_hat!(op, ep)
#         # fix_inequalities_nonhat!(op, ep)
#
#         @update op.δq   fδq       Δ ψ verb  op.qh0 op.δqh op.ρh
#         if !fixρ
#             @update ep.ρ   fρ     Δ ψ verb  op.qh0 op.δqh op.ρh
#         end
#
#
#         verb > 1 && println(" Δ=$Δ\n")
#         verb > 1 && println(all_therm_func(op, ep))
#
#         @assert isfinite(Δ)
#         ok = ok && Δ < ϵ
#         ok && break
#     end
#
#     ok
# end


function converge(;
        q0 = 1.,
        δq=0.5,
        qh0=0., δqh=0.6,
        ρ=0, ρh=0,
        α=0.1, Δ=0.1,
        ϵ=1e-4, maxiters=100000, verb=3, ψ=0.,
        fixρ=true, fixnorm=true, extrap=-1
    )
    op = OrderParams(q0,δq,qh0,δqh,ρh)
    ep = ExtParams(α, ρ, Δ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixρ=fixρ, fixnorm=fixnorm,extrap=extrap)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
        # q0=2.0860295957019495,δq=7.816340972218491,qh0=0.034142290508436736,δqh=0.12793710043539372,ρh=0.0012793710043539372,
        # q0=0.9805369500611506,δq=2.450455809444866,qh0=0.003281540090095331,δqh=0.40808734283053366,ρh=0.40400646940222834,
        q0 = 1, ρ=1,
        qh0 = 0.08773765511622797,
  δqh = 100000,
  ρh = 100000,
  δq = 0.00001,
        # ρ=0.01, α=2., Δ=0.02,
        α=2., Δ=0.02,
        ϵ=1e-4, maxiters=10000,verb=3, ψ=0., fixnorm=false,
        kws...)

    op = OrderParams(q0,δq,qh0,δqh,ρh)
    ep = ExtParams(first(α), first(ρ), Δ)
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; ρ=ρ, α=α, fixnorm=fixnorm, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
    α=1, ρ=1, Δ=0.1,
    resfile = "results_RS_noisy_UNSAT_unconstrained.txt",
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

# function span(;
#     δq=0.3188,
#     qh0=0.36889,δqh=0.36889, ρh=0.56421,
#     ρ=0.384312, α=1,
#     ϵ=1e-6, maxiters=10000,verb=2, ψ=0.,
#     kws...)
#
#     op = OrderParams(δq,qh0,δqh,ρh)
#     ep = ExtParams(first(α), first(ρ))
#     pars = Params(ϵ, ψ, maxiters, verb)
#     return span!(op, ep, pars; ρ=ρ,α=α, kws...)
# end
#
# function span!(op::OrderParams, ep::ExtParams, pars::Params;
#     α=1, ρ=1,
#     resfile = "results.txt",
#     fixm=false, fixρ=false, mlessthan1=false)
#
#     if !isfile(resfile)
#         open(resfile, "w") do f
#             allheadersshow(f, ExtParams, OrderParams, ThermFunc)
#         end
#     end
#
#     results = []
#     for α in α, ρ in ρ
#         fixρ && (ep.ρ = ρ)
#         ep.α = α;
#
#         ok = converge!(op, ep, pars; fixρ=fixρ)
#         tf = all_therm_func(op, ep)
#         push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))
#         if ok
#             open(resfile, "a") do rf
#                 println(rf, plainshow(ep), " ", plainshow(op), " ", plainshow(tf))
#             end
#         end
#         !ok && break
#         pars.verb > 0 && print(ep, "\n", tf,"\n")
#     end
#     return results
# end
#
#
# function readparams(file, line = 0)
#     res = readdlm(file)
#     line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
#     ep = ExtParams(res[line,1:2]...)
#     op = OrderParams(res[line,3:6]...)
#     tf = ThermFunc(res[line,end])
#     return ep, op, tf
# end

end ## module
