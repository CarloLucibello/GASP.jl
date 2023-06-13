module Perc

using LittleScienceTools.Roots
using QuadGK
using AutoGrad
include("../common.jl")


###### INTEGRATION ######
const ∞ = 30.0
const dx = 0.1
# const dx = 0.02
const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-8, maxevals=10^7)[1]

# function deriv(f::Function, i::Integer, x...; δ::Float64 = 1e-5)
#     f0 = f(x[1:i-1]..., x[i]-δ, x[i+1:end]...)
#     f1 = f(x[1:i-1]..., x[i]+δ, x[i+1:end]...)
#     return (f1-f0) / 2δ
# end

# Numerical Derivative
# Can take also directional derivative
# (tune the direction with i and δ).
function deriv(f::Function, i, x...; δ = 1e-7)
    x1 = deepcopy(x) |> collect
    x1[i] .+= δ
    f0 = f(x...)
    f1 = f(x1...)
    return (f1-f0) / vecnorm(δ)
end

deriv(f::Function, i::Integer, x...) = grad(f, i)(x...)

############### PARAMS

mutable struct OrderParams
    q0::Float64
    q1::Float64
    qh0::Float64
    qh1::Float64
    qh2::Float64
    ρh::Float64
    m::Float64 # parisi breaking parameter
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
    Σ::Float64
    F::Float64
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
Gi(q0,q1,qh0,qh1,qh2,ρh,m,ρ) = 0.5*(qh0*q0*m + qh1*q1*(1-m) - qh2 - 2ρ*ρh)

∂m_Gi(q0,q1,qh0,qh1,qh2,ρh,m,ρ) = 0.5*(q0*qh0 - qh1*q1)

#### ENTROPIC TERM ####

Gs(qh0,qh1,qh2,ρh,m) = ((qh0 + Power(ρh,2))/(m*qh0 - (-1 + m)*qh1 - qh2) -
    (1 - 1/m)*Log(qh1 - qh2) - Log(m*qh0 - (-1 + m)*qh1 - qh2)/m)/2

∂m_Gs(qh0,qh1,qh2,ρh,m) =  (- (qh0 - qh1)/(m*(m*qh0 - (-1 + m)*qh1 - qh2)) -
    ((qh0 - qh1)*(qh0 + Power(ρh,2)))/Power(m*qh0 - (-1 + m)*qh1 - qh2,2) -
    Log(qh1 - qh2)/Power(m,2) + Log(m*qh0 - (-1 + m)*qh1 - qh2)/Power(m,2))/2

#### ENERGETIC TERM ####

function argGe(q0, q1, expβ, z0, z1)
    @assert q1-q0 > 0
    @assert q0 > 0
    @assert q1 < 1
    c = (√(q1-q0)*z1 + √q0*z0)/√(1-q1)
    h = H(-c)
    h + expβ*(1-h)
end

function Ge(q0, q1, m, β, ρ)
    expβ = exp(-2β)
    2∫D(z0->begin
        l = log(∫D(z1 -> argGe(q0, q1, expβ, z0, z1)^m))
        # l * H(-√(ρ^2/(q0-ρ^2))*z0) # TECHER STUDENT
        l/2 # RANDOM
    end)/m
end

# ∂q0_Ge(q0, q1, m, β, ρ) = deriv(Ge, 1, q0, q1, m, β, ρ) #num deriv unstable when q0≈q1
∂q0_Ge(q0, q1, m, β, ρ) = deriv(Ge, [1,2], q0, q1, m, β, ρ) - deriv(Ge, 2, q0, q1, m, β, ρ)
∂q1_Ge(q0, q1, m, β, ρ) = deriv(Ge, 2, q0, q1, m, β, ρ)
∂m_Ge(q0, q1, m, β, ρ) = deriv(Ge, 3, q0, q1, m, β, ρ)
∂β_Ge(q0, q1, m, β, ρ) = deriv(Ge, 4, q0, q1, m, β, ρ)
∂ρ_Ge(q0, q1, m, β, ρ) = deriv(Ge, 5, q0, q1, m, β, ρ)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 q1 qh0 qh1 qh2 ρh m
    @extract ep: α β ρ
    Gi(q0,q1,qh0,qh1,qh2,ρh,m,ρ) + Gs(qh0,qh1,qh2,ρh,m) + α*Ge(q0, q1, m, β, ρ)
end

## Thermodinamic functions
# The energy of the pure states selected by x
# E = -∂(m*ϕ)/∂m
# if working at fixed x. If x is optimized Σ=0 and
# E = -ϕ
# so this formula is valid both at fixed and at optimized x
function all_therm_func(op::OrderParams, ep::ExtParams)
    @extract op: m
    @extract ep: β
    ϕ = free_entropy(op, ep)
    Σ = -m^2*im_fun(op, ep, m)
    F = (m*ϕ - Σ) /(-β*m)
    # E = ???
    return ThermFunc(ϕ, Σ, F)
end

###########################
fqh0(q0, q1, m, α, β, ρ) = -2/m * α * ∂q0_Ge(q0, q1, m, β, ρ)
fqh1(q0, q1, m, α, β, ρ) = 2α/(m-1) * ∂q1_Ge(q0, q1, m, β, ρ)
fρh(q0, q1, m, α, β, ρ) = α * ∂ρ_Ge(q0, q1, m, β, ρ)

fq0(qh0,qh1,qh2,ρh,m) = (qh0 + ρh^2)/Power(m*qh0 - (-1 + m)*qh1 - qh2,2)
fq1(qh0,qh1,qh2,ρh,m) = (-(m*Power(qh0 - qh1,2) - (qh1 - qh2)*(qh1 + Power(ρh,2)))/
    ((qh1 - qh2)*Power(m*(qh0 - qh1) + qh1 - qh2,2)))

fρ(qh0,qh1,qh2,ρh,m) = ρh/(qh1-qh2 + m*(qh0 - qh1))

iρh(ρ,qh0,qh1,qh2,m) = (true, ρ*(qh1-qh2 + m*(qh0 - qh1)))

function iqh2_fun(qh0, qh1, qh2, ρh, m)
    (-1 + (1 - 1/m)/(qh1 - qh2) + 1/(m*(m*qh0 - (-1 + m)*qh1 - qh2)) +
     (qh0 + Power(ρh,2))/Power(m*qh0 - (-1 + m)*qh1 - qh2,2))
end

function iqh2(qh0, qh1, qh2₀, ρh, m, atol=1e-8)
    ok, qh2, it, normf0 = findroot(qh2 -> iqh2_fun(qh0, qh1, qh2, ρh, m), qh2₀, NewtonMethod(atol=atol))
    #ok, M, it, normf0 = findzero_interp(M->∂_Ge(5, Q, q0, M, x, K, avgξ, varξ, f′), M0, dx=0.1)

    ok || @warn("iqh2 failed: iqh2=$(qh2), it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, qh2
end

function im_fun(op::OrderParams, ep::ExtParams, m)
    @extract op: q0 q1 qh0 qh1 qh2 ρh
    @extract ep: α ρ β
    ∂m_Gi(q0,q1,qh0,qh1,qh2,ρh,m,ρ) + ∂m_Gs(qh0,qh1,qh2,ρh,m) + α*∂m_Ge(q0, q1, m, β, ρ)
end

function im(op::OrderParams, ep::ExtParams, m₀, atol=1e-6)
    ok, m, it, normf0 = findroot(m -> im_fun(op, ep, m), m₀, NewtonMethod(atol=atol))
    ok || @warn("im failed: m=$m, it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, m
end

###############################


function fix_inequalities_hat!(op,ep)
    ok = false
    t = 0
    while !ok
        t += 1
        ok = true
        if op.qh1 < op.qh0
            mid = 0.5*(op.qh0 + op.qh1)
            op.qh0 = mid - 1e-8
            op.qh1 = mid + 1e-8
            ok = false
        end

        if op.qh1 < op.qh2
            op.qh1 += 1e-4
            ok = false
        end

        if op.qh1 -  op.qh2 +  op.m * ( op.qh0 -  op.qh1) < 0
            op.qh1 += 1e-4
            ok = false
        end
        t > 10000 && (println("fix NO SOL"); break)
    end
    t > 1 && println("***fixed hat***")
    return t > 1
end


function fix_inequalities_nonhat!(op, ep)
    ok = false
    t = 0
    while !ok
        ok = true
        t += 1
        if op.q0 < 0
            op.q0 = rand() * 1e-5
            ok = false
        end
        if op.q1 < 0
            op.q1 = rand() * 1e-5
            ok = false
        end
        if op.q1 + 1e-9 > 1
            op.q1 -= 1e-9
            ok = false
        end
        if op.q1 < op.q0 + 1e-9
            # q0 -= 1e-5
            op.q1 += 1e-5
            ok = false
        end
        if ep.ρ^2 / op.q0  + 1e-7 > 1
            op.q0 += 1e-7
            ok = false
        end
        t > 10000 && (println("fix NO SOL"); break)
    end
    t > 1 && println("***fixed nonhat***")
    return t > 1
end


function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixm = false, fixρ = true)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    it = 0
    for it = 1:maxiters
        Δ = 0.0
        verb > 1 && println("it=$it")

        fix_inequalities_hat!(op, ep)
        fix_inequalities_nonhat!(op, ep)

        @update  op.qh0    fqh0       Δ ψ verb  op.q0 op.q1 op.m ep.α ep.β ep.ρ
        @update  op.qh1    fqh1       Δ ψ verb  op.q0 op.q1 op.m ep.α ep.β ep.ρ
        @updateI op.qh2 ok   iqh2     Δ ψ verb  op.qh0 op.qh1 op.qh2 op.ρh op.m
        if fixρ
            @updateI op.ρh ok   iρh   Δ ψ verb  ep.ρ op.qh0 op.qh1 op.qh2 op.m
        else
            @update  op.ρh  fρh       Δ ψ verb  op.q0 op.q1 op.m ep.α ep.β ep.ρ
        end

        fix_inequalities_hat!(op, ep)
        fix_inequalities_nonhat!(op, ep)

        @update op.q0   fq0       Δ ψ verb  op.qh0 op.qh1 op.qh2 op.ρh op.m
        @update op.q1   fq1       Δ ψ verb  op.qh0 op.qh1 op.qh2  op.ρh op.m
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op.qh0 op.qh1 op.qh2  op.ρh op.m
        end

        if !fixm && it%10==0
            @updateI op.m ok   im    Δ ψ verb  op ep op.m
        end

        verb > 1 && println(" Δ=$Δ\n")

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end

    if verb > 0
        println(ok ? "converged" : "failed", " (it=$it Δ=$Δ)")
        println(op)
    end
    ok
end

function converge(;
        q0=0.1, q1=0.5,
        qh0=0., qh1=0., qh2 =0.6,
        m = 1.00001, ρ=0, ρh=0,
        α=0.1, β=1,
        ϵ=1e-6, maxiters=100000, verb=2, ψ=0.,
        fixm = false, fixρ=true
    )
    op = OrderParams(q0,q1,qh0,qh1,qh2, ρh,m)
    ep = ExtParams(α, β, ρ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixm=fixm, fixρ=fixρ)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end

function span(;
        q0=0.3188,q1=0.31885,
        qh0=0.36889,qh1=0.36889,qh2=-1.09921, ρh=0.56421,
        m=0.1, ρ=0.384312, α=1, β=1,
        ϵ=1e-6, maxiters=10000,verb=2, ψ=0.,
        kws...)

    op = OrderParams(q0,q1,qh0,qh1,qh2,ρh, first(m))
    ep = ExtParams(first(α), first(β), first(ρ))
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; ρ=ρ,β=β,α=α,m=m, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
        m=0,  α=1, β=1.6, ρ=1,
        resfile = "results.txt",
        fixm=false, fixρ=false, mlessthan1=false)

    if !isfile(resfile)
        open(resfile, "w") do f
            allheadersshow(f, ExtParams, OrderParams, ThermFunc)
        end
    end

    results = []
    for m in m, α in α, β in β, ρ in ρ
        fixm && (op.m = m)
        fixρ && (ep.ρ = ρ)
        ep.α = α; ep.β = β;

        opold, epold = deepcopy(op), deepcopy(ep)
        ok = converge!(op, ep, pars; fixm=fixm, fixρ=fixρ)
        tf = all_therm_func(op, ep)
        tf.Σ < -1e-7 && @warn("Sigma negative")
        # op.m > 1+1e-6 && @warn("m > 1: $(op.m)")
        if op.m > 1  && mlessthan1 && !fixm
            op, ep = opold, epold
            op.m = 1.0000001
            ok = converge!(op, ep, pars; fixm=true, fixρ=fixρ)
            tf = all_therm_func(op, ep)
        end
        push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))
        if ok
            open(resfile, "a") do rf
                println(rf, plainshow(ep), " ", plainshow(op), " ", plainshow(tf))
            end
        end
        ok || @warn("!ok")
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end

function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    ep = ExtParams(res[line,1:3]...)
    op = OrderParams(res[line,4:10]...)
    tf = ThermFunc(res[line,11:end]...)
    return ep, op, tf
end

end # module
