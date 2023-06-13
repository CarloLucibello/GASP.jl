module PhaseRetr

using LittleScienceTools.Roots
using QuadGK
using FastGaussQuadrature


include("../common.jl")


###### INTEGRATION  ######
const ∞ = 12.0
const dx = .05

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-10, reltol=1e-10, maxevals=5*10^3)[1]

function uneven_spacings(; β=1., intervals=5, ∞=5.)
    y = Float64[];
    x = 0;
    shift = 0;
    for i = floor(Int, -intervals/2):floor(intervals/2)
        x += (2^β)^abs(i)
        push!(y, x)
        i == 0 && (shift = x - 0.5)
    end
    y -= shift
    scale!(y, ∞/maximum(y))
    return y
end

∫D_centered(f, center, β; int=interval) = quadgk(z->begin
r = G(z) * f(z)
isfinite(r) ? r : 0.0
end, (int + center)..., abstol=1e-12, reltol=1e-12, maxevals=10^4)[1]


let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    global gw
    gw(n::Int) = get!(s, n) do
        (x,w) = gausshermite(n)
        return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
    end
end



function ∫DD(f; n=151)
    (xs, ws) = gw(n)
    s = 0.0
    for (x,w) in zip(xs, ws)
        y = f(x)
        s += w  * ifelse(isfinite(y), y, 0.0)
    end
    return s
end


function deriv(f::Function, i::Integer, x...; δ::Float64 = 1e-7)
    try
        f0 = f(x[1:i-1]..., x[i]-δ, x[i+1:end]...)
        f1 = f(x[1:i-1]..., x[i]+δ, x[i+1:end]...)
        return (f1-f0) / 2δ
    catch
        try
            f0 = f(x[1:i-1]..., x[i]-δ, x[i+1:end]...)
            f1 = f(x[1:i-1]..., x[i], x[i+1:end]...)
            return (f1-f0) / δ
        catch
            f0 = f(x[1:i-1]..., x[i], x[i+1:end]...)
            f1 = f(x[1:i-1]..., x[i]+δ, x[i+1:end]...)
            return (f1-f0) / δ
        end
    end
end

############### PARAMS

mutable struct OrderParams
    q0::Float64
    qh0::Float64
    qh1::Float64
    ρh::Float64
end

mutable struct ExtParams
    α::Float64
    β::Float64
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
    E::Float64
    S::Float64
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

Gi(ρ, q0, ρh, qh0, qh1) = -0.5 * (2 * ρ * ρh + qh1 - q0 * qh0)

#### ENTROPIC TERM ####
Gs(ρh, qh0, qh1) = 0.5 * ((qh0 + ρh^2)/(qh0 - qh1) - log(qh0 - qh1))

∂ρh_Gs(ρh, qh0, qh1) = ρh / (qh0 - qh1)
∂qh0_Gs(ρh, qh0, qh1) = -0.5 * ((ρh^2 + qh0)/(qh0 - qh1)^2)
∂qh1_Gs(ρh, qh0, qh1) = 0.5 * ((ρh^2 + qh0)/(qh0 - qh1)^2 + 1/(qh0 - qh1))

#### ENERGETIC TERM ####

fy(ρ, q0, u0, z0) = (sqrt(1 - ρ^2 / q0) * u0 + ρ / sqrt(q0) * z0)^2
exp_argGe(y, q0, z0, u) = 1/2 * (y - (u * √(1-q0) + z0 * √(q0))^2)^2

function inner_integral(y, q0, z0, β)
    ∫D(u->begin
       exp(- β * exp_argGe(y, q0, z0, u))
   end)
end

function Ge(q0, ρ, β)
    ∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            log(inner_integral(y, q0, z0, β))
        end)
    end)
end

function ∂q0_Ge_an(q0, ρ, β)
    ∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            num = ∫D(u->begin
            exp(- β * exp_argGe(y, q0, z0, u)) * β * (y - (u*√(1-q0) + z0*√(q0))^2) * 2 *
            ((u*√(1-q0) + z0*√(q0)) * (-1/(2*√(1-q0))*u + 1/(2*√(q0))*z0) -
            (√(1-ρ^2/q0)*u0 + ρ/√(q0) * z0)*(1/(2*√(1-ρ^2/q0))*ρ^2/q0^2*u0 - ρ/(2*q0^(3/2))*z0)) end)
            den = inner_integral(y, q0, z0, β)
            num / den
        end)
    end)
end

function ∂q0_Ge(q0, ρ, β)
    try
        ∂q0_Ge_an(q0, ρ, β)
    catch
        deriv(Ge, 1, q0, ρ, β)
    end
end

function ∂β_Ge_an(q0, ρ, β)
    ∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            num = - ∫D(u->begin
                exp(- β * exp_argGe(y, q0, z0, u)) * exp_argGe(y, q0, z0, u)  end)
            den = inner_integral(y, q0, z0, β)
            num / den
        end)
    end)
end

∂ρ_Ge(q0, ρ, β) = deriv(Ge, 2, q0, ρ, β)
∂β_Ge(q0, ρ, β) = deriv(Ge, 3, q0, ρ, β)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 qh0 qh1 ρh
    @extract ep: α β ρ
    Gi(ρ, q0, ρh, qh0, qh1) + Gs(ρh, qh0, qh1) + α*Ge(q0, ρ, β)
end

function energy(op::OrderParams, ep::ExtParams)
    @extract op: q0
    @extract ep: α β ρ
    return - α * ∂β_Ge_an(q0, ρ, β)
end

## Thermodinamic functions
function all_therm_func(op::OrderParams, ep::ExtParams)
    ϕ = free_entropy(op, ep)
    E = energy(op, ep)
    S = (ep.β * E + ϕ)
    return ThermFunc(ϕ, E, S)
end

###########################
#### SADDLE POINT EQUATIONS ####
#
# q00 = 1
# q1 = 1
# q0 = - 2 * ∂qh0_Gs
# qh00 = -1
#
# qh0 = -2α * ∂q0_Ge
# 0 = -1/2 + ∂qh1_Gs --->>> qh1
# 0 = -ρ + ∂ρh_Gs  --->>> ρh
#

fqh0(q0, ρ, α, β) = -2 * α * ∂q0_Ge(q0, ρ,β)
fρh(q0, ρ, α, β) = α * ∂ρ_Ge(q0, ρ, β)

fq0(ρh, qh0, qh1) = -2 * ∂qh0_Gs(ρh, qh0, qh1)
fρ(ρh, qh0, qh1) = ρh / (qh0 - qh1)


iρh(ρ, qh0, qh1) = (true, ρ*(qh0 - qh1))

function iqh1_fun(qh0, qh1, ρh)
    -0.5 + ∂qh1_Gs(ρh, qh0, qh1)
end

function iqh1(qh0, qh1₀, ρh, atol=1e-12)
    ok, qh1, it, normf0 = findroot(qh1 -> iqh1_fun(qh0, qh1, ρh), qh1₀, NewtonMethod(atol=atol))

    ok || @warn("iqh1 failed: iqh1=$(qh1), it=$it, normf0=$normf0")
    return ok, ok ? qh1 : qh1₀
end

###############################

function fix_inequalities_hat(ρh, qh0, qh1)
    ok = false
    t = 0
    while !ok
        t += 1
        ok = true
        if qh0 < qh1
            qh0 += 1e-4 * rand()
            qh1 -= 1e-4 * rand()
            ok = false
        end
    end
    t > 1 && println("***fixed***")
    return t, ρh, qh0, qh1
end

function fix_inequalities_nonhat(ρ, q0)
    ok = false
    t = 0
    while !ok
        ok = true
        t += 1
        if q0 < 0
            q0 = rand() * 1e-4
            ok = false
        end
        if q0 > 1
            q0 = 1 - rand() * 1e-4
            ok = false
        end
        if ρ < 0
            ρ = rand() * 1e-4
            ok = false
        end
        if ρ > 1
            ρ = 1 - rand() * 1e-4
            ok = false
        end
        if ρ^2 / q0 > 1
            q0 += 1e-4 * rand()
            # ρ -= 1e-4 * rand()
            ok = false
        end
    end
    t > 1 && println("***fixed***")
    return t, ρ, q0
end

###############################

function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixρ = true)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    t, op.ρh, op.qh0, op.qh1 = fix_inequalities_hat(op.ρh, op.qh0, op.qh1)
    t, ep.ρ, op.q0 = fix_inequalities_nonhat(ep.ρ, op.q0)


    it = 0
    for it = 1:maxiters
        @time begin
            Δ = 0.0
            verb > 1 && println("it=$it")
            qh0 = 0
            qh1 = 0

            @update  op.qh0      fqh0       Δ ψ verb  op.q0 ep.ρ ep.α ep.β
            @updateI op.qh1   ok iqh1       Δ ψ verb  op.qh0 op.qh1 op.ρh ϵ
            if fixρ
                @updateI op.ρh  ok iρh      Δ ψ verb  ep.ρ op.qh0 op.qh1
            else
                @update  op.ρh   fρh        Δ ψ verb  op.q0 ep.ρ ep.α
            end
            t, op.ρh, op.qh0, op.qh1 = fix_inequalities_hat(op.ρh, op.qh0, op.qh1)
            t > 1 && (ok = false)

            @update op.q0        fq0        Δ ψ verb  op.ρh op.qh0 op.qh1
            if !fixρ
                @update ep.ρ     fρ         Δ ψ verb  op.ρh op.qh0 op.qh1
            end
            t, ep.ρ, op.q0 = fix_inequalities_nonhat(ep.ρ, op.q0)
            t > 1 && (ok = false)

        end

        verb > 1 && println(" Δ=$Δ\n")

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end
end

function converge(;q0=0.30322682066687645,qh0=0.5421606786525736,qh1=-0.8929950095924145,ρh=0.28703123973704064,
        α=1.2,β=5.0,ρ=0.2,
        ϵ=1.0e-4,ψ=0.0,maxiters=100000,verb=2, fixρ=true
    )
    op = OrderParams(q0, qh0, qh1, ρh)
    ep = ExtParams(α, β, ρ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixρ=fixρ)
    tf = all_therm_func(op, ep)
    println(tf)
    println()
    return op, ep, pars, tf
end

function initialize_op(;q0=0.1876445158451444,qh0=0.22372659959912677,qh1=-1.0072522502432357,ρh=0.24619588717419003,
        α=1.2, β=1., ρ=0.2,
        resfile=nothing)

    op = 0
    if resfile == nothing
        op = OrderParams(q0, qh0, qh1, ρh)
    else
        a = readdlm(resfile)
        a1 = map(i->a[i,1],1:size(a,1))
        a2 = map(i->a[i,2],1:size(a,1))
        a3 = map(i->a[i,3],1:size(a,1))
        l = a[findmin(map(i->abs(α-a1[i]) + abs(β-a2[i])/β + abs(ρ-a3[i]), 1:length(a1)))[2],:]
        op = OrderParams(l[7], l[8], l[9], l[10])
    end
    return op
end

function span(; lstα = [0.6], lstρ = [0.6], lstβ=[15.], op = nothing,
                ϵ = 1e-5, ψ = 0.1, maxiters = 500, verb = 4,
                resfile = nothing)

    default_resfile = "results_phase_retrieval_RS_finiteT.txt"
    resfile == nothing && (resfile = default_resfile)

    pars = Params(ϵ, ψ, maxiters, verb)
    results = Any[]

    lockfile = "reslock.tmp"

    op == nothing && (op = initialize_op())
    ep = ExtParams(lstα[1], lstβ[1], lstρ[1])
    println(op)

    for α in lstα, β in lstβ, ρ in lstρ
        op = initialize_op(resfile=resfile, α=α, β=β, ρ=ρ)
        ep.α = α
        ep.β = β
        ep.ρ = ρ

        println()
        println("NEW POINT")
        println("       α=$(ep.α) β=$(ep.β) ρ=$(ep.ρ)")
        println()

        converge!(op, ep, pars; fixρ=true)
        tf = all_therm_func(op, ep)
        println(tf)

        push!(results, (deepcopy(op), deepcopy(ep), deepcopy(tf)))

        open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
    end
    return results, op
end

end ## module
