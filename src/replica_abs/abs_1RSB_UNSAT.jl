module PhaseRetr

# using LittleScienceTools.Roots
using QuadGK
using AutoGrad
using Cubature
using ForwardDiff
using IterTools: product
using NLsolve: nlsolve
using Distributed
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 12.0
const dx = 0.01


using FastGaussQuadrature

let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    global gw
    gw(n::Int) = get!(s, n) do
    (x,w) = gausshermite(n)
    return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
    end
end

function ∫DD(f; n=141)
    (xs, ws) = gw(n)
    s1 = 0.0
    for (x,w) in zip(xs, ws)
        y = f(x)
        s1 += w  * ifelse(isfinite(y), y, 0.0)
    end
    (xs, ws) = gw(n+1)
    s2 = 0.0
    for (x,w) in zip(xs, ws)
        y = f(x)
        s2 += w  * ifelse(isfinite(y), y, 0.0)
    end
    return (s1 + s2)/2
end



const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) .* f(z)
        isfinite(r) ? r : 0.0
    end, int..., atol=1e-9, maxevals=10^9)[1]

# ∫∫D(f) = ∫D(x->∫D(y->f(x,y)))

## Cubature.jl
∫∫D(f, xmin::Vector, xmax::Vector) = hcubature(z->begin
            r = G(z[1])*G(z[2])*f(z[1],z[2])
            isfinite(r) ? r : 0.0
        end, xmin, xmax, abstol=1e-7)[1]


∫∫D(fdim, f, xmin::Vector, xmax::Vector) = hcubature(fdim, (z,y)->begin
        y .= (G(z[1]).*G(z[2])).*f(z[1],z[2])
        @. isfinite(y) ? y : 0.0
    end, xmin, xmax, abstol=1e-7)[1]

function ∫∫D(f)
    ints = [(interval[i],interval[i+1]) for i=1:length(interval)-1]
    intprods = product(ints, ints)
    fdim = length(f(0.,0.))
    sum(ip-> begin
            xmin = [ip[1][1],ip[2][1]]
            xmax = [ip[1][2],ip[2][2]]
            if fdim==1
                ∫∫D(f, xmin, xmax)
            else
                ∫∫D(fdim, f, xmin, xmax)
            end
        end, intprods)
end


# Numerical Derivative
# Can take also directional derivative
# (tune the direction with i and δ).
function deriv(f::Function, i, x...; δ = 1e-5)
    x1 = deepcopy(x) |> collect
    x1[i] .+= δ
    f0 = f(x...)
    f1 = f(x1...)
    return (f1-f0) / vecnorm(δ)
end

# Numerical Derivative for member of the structured input
function deriv_(f::Function, i::Int, x...; arg=1, δ=1e-5)
    x1 = deepcopy(x)
    setfield!(x1[arg], i, getfield(x1[arg], i) + δ)
    f0 = f(x...)
    f1 = f(x1...)
    return (f1-f0) / δ
end


############### PARAMS ################

mutable struct OrderParams
    ρ::Float64
    q0::Float64
    q1::Float64
    δq::Float64
    ρh::Float64
    qh0::Float64
    qh1::Float64
    δqh::Float64
    m::Float64 # parisi breaking parameter
end
collect(op::OrderParams) = [getfield(op, f) for f in fieldnames(op)]

mutable struct ExtParams
    α::Float64
    λ::Float64
    Δ::Float64 #noise variance
end

mutable struct Params
    ϵ::Float64 # stop criterium
    ψ::Float64 # dumping
    maxiters::Int
    verb::Int
    solvenl::Int
end

mutable struct ThermFunc
    ϕ::Float64
    Σ::Float64
    E::Float64
end

Base.show(io::IO, op::OrderParams) = shortshow(io, op)
Base.show(io::IO, ep::ExtParams) = shortshow(io, ep)
Base.show(io::IO, tf::ThermFunc) = shortshow(io, tf)

###################################################################################

# Mathematica compatibility
Power(x,y) = x^y
Log(x) = log(x)

#### INTERACTION TERM ####
function Gi(op, ep)
    @extract op: q0 q1 δq qh0 qh1 δqh ρh m ρ
    (q1*δqh - 2*ρ*ρh - δq*qh1 + q0*qh0*m - q1*qh1*m)/2
end

function ∂m_Gi(op, ep)
    @extract op: q0 q1 δq qh0 qh1 δqh ρh ρ
    (q0*qh0 - q1*qh1) / 2
end

#### ENTROPIC TERM ####

function Gs(op, ep)
    @extract op: qh0 qh1 δqh ρh m
    @extract ep: λ
    0.5*((Power(ρh,2) + qh0)/(δqh+λ + (qh0 - qh1)*m) + Log(δqh+λ)/m - Log(δqh+λ + (qh0 - qh1)*m)/m)
end

function ∂m_Gs(op)
    @extract op: qh0 qh1 δqh ρh m
    @extract ep: λ
    (-((qh0 - qh1)/(m*(δqh+λ + m*(qh0 - qh1)))) -
    ((qh0 - qh1)*(qh0 + Power(ρh,2)))/Power(δqh+λ + m*(qh0 - qh1),2) -
    Log(δqh+λ)/Power(m,2) + Log(δqh+λ + m*(qh0 - qh1))/Power(m,2))/2
end

#### ENERGETIC TERM ####

function argGe(y, h, δq)
    ### find min of 1/2 u^2 + (y - abs(√δq*u + h))^2
    # u* = 2√δq*(y-|h|)*sign(h) / (1+2δq)
    (y - abs(h))^2 / (1 + 2δq)
end
function ∂z_argGe(y, h, δq)
    -2sign(h)*(y - abs(h)) / (1 + 2δq)
end
function ∂δq_argGe(y, h, δq)
    -2*(y - abs(h))^2 / (1 + 2δq)^2
end

function Ge0(y, h, q10, δq, m)
    return log(∫D(z-> exp(-m*argGe(y, √(q10)*z+h, δq))))/m
end

function Ge₀(y, h, q10, δq, m)
    @assert 1 + 2*m*q10 + 2*δq > 0
    a = sqrt((1 + 2*δq)*(1 + 2*m*q10 + 2*δq))
    Z1 = H(-((2*m*q10*y - h*(1 + 2*δq))/
        (sqrt(q10)*a)))*exp(-(m*Power(h + y,2))/(1 + 2*m*q10 + 2*δq))
    Z2 = H(-((2*m*q10*y + h*(1 + 2*δq))/
        (sqrt(q10)*a)))*exp(-(m*Power(h - y,2))/(1 + 2*m*q10 + 2*δq))
    Z = (1 + 2*δq)*(Z1 + Z2) / a
    1/m * log(Z)
end

∂h_Ge₀ = (y, h, q10, δq, m) -> begin
            ForwardDiff.derivative(h->Ge₀(y, h, q10, δq, m),h)
        end
 ∂δq_Ge₀ = (y, h, q10, δq, m) -> begin
            ForwardDiff.derivative(δq->Ge₀(y, h, q10, δq, m),δq)
        end
∂q10_Ge₀ = (y, h, q10, δq, m) -> begin
            ForwardDiff.derivative(q10->Ge₀(y, h, q10, δq, m),q10)
        end



function Ge(op, ep)
    @extract op: ρ q0 q1 δq m

    ∫∫D((u0,z0)->begin
        Ge₀(abs(u0), ρ*u0 + √(q0 - ρ^2)*z0, q1-q0, δq, m)
    end)
end

function Ge2(op, ep)
    @extract op: ρ q0 q1 δq m
    ### m = ∞ asymptote with divergence
    # 1/m*∫∫D((u0,z0)->begin
    #    log(2*cosh(2m*u0*(ρ*u0+√(q0-ρ^2)*z0)/(1+2δq+2m*(q1-q0))))
    #       end) + 1/(2m)*log((1+2δq)/(1+2δq+2m*(q1-q0))) - (1+q0)/(1+2δq+2m*(q1-q0))

    ### m = ∞ limit without the divergence
    ∫∫D((u0,z0)->begin
        log(2*cosh((u0*(ρ*u0+√(q0-ρ^2)*z0))/(q1-q0)))
            end) + 1/2*log((1+2*δq)/(2*(q1-q0))) - (1+q0)/(2*(q1-q0))
end


function ∂q0_Ge(op, ep)
    @extract op: q0 q1 ρ δq m
    -0.5*m*∫∫D((z0,u0)->begin
    # -0.5*m*∫DD(z0->∫DD(u0->begin

        ∂h_Ge₀(abs(u0), ρ*u0 + √(q0 - ρ^2)*z0, q1-q0, δq, m)^2
    end)
end
function ∂δq_Ge(op, ep)
    @extract op: q0 q1 ρ δq m
    ∫∫D((z0,u0)->begin
    # ∫DD(z0->∫DD(u0->begin
        ∂δq_Ge₀(abs(u0), ρ*u0 + √(q0 - ρ^2)*z0, q1-q0, δq, m)
    end)
end
function ∂q1_Ge(op, ep)
    @extract op: q0 q1 ρ δq m
    ∫∫D((z0,u0)->begin
    # ∫DD(z0->∫DD(u0->begin
        ∂q10_Ge₀(abs(u0), ρ*u0 + √(q0 - ρ^2)*z0, q1-q0, δq, m)
    end)
end
function ∂ρ_Ge(op, ep)
    @extract op: q0 q1 ρ δq m
    ∫∫D((z0,u0)->begin
    # ∫DD(z0->∫DD(u0->begin
        ((u0 - ρ/√(q0 - ρ^2)*z0)*
        ∂h_Ge₀(abs(u0), ρ*u0 + √(q0 - ρ^2)*z0, q1-q0, δq, m))
    end)
end



# ∂q0_Ge(op,ep) = deriv_(Ge, 2, op, ep)
# ∂q1_Ge(op,ep) = deriv_(Ge, 3, op, ep)
# ∂δq_Ge(op,ep) = deriv_(Ge, 4, op, ep)
# ∂ρ_Ge(op,ep) = deriv_(Ge, 1, op, ep)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    Gi(op, ep) + Gs(op, ep) + ep.α*Ge(op, ep)
end

∂m_ϕ(op,ep) = deriv_(free_entropy, 9, op, ep)


## Thermodinamic functions
# The energy of the pure states selected by x
# E = -∂(m*ϕ)/∂m
# if working at fixed x. If x is optimized Σ=0 and
# E = -ϕ
# so this formula is valid both at fixed and at optimized x
function all_therm_func(op::OrderParams, ep::ExtParams)
    ϕ = free_entropy(op, ep)
    E = -ϕ - op.m * ∂m_ϕ(op, ep)
    Σ = op.m*(ϕ + E)
    return ThermFunc(ϕ, Σ, E)
end

#################  SADDLE POINT  ##################
fqh0(op, ep) = -2/op.m * ep.α * ∂q0_Ge(op, ep)
fqh1(op, ep) = 2ep.α * ∂δq_Ge(op, ep)
fδqh(op, ep) = op.qh1*op.m -2ep.α * ∂q1_Ge(op, ep)
fρh(op, ep) = ep.α * ∂ρ_Ge(op, ep)

fm(op, ep) = op.m

function fq0(op, ep)
    @extract op: qh0 qh1 δqh ρh m
    @extract ep: λ
    (qh0 + ρh^2) / (δqh+λ + m*(qh0 - qh1))^2
end

function fδq(op, ep)
    @extract op: qh0 qh1 δqh ρh m q1
    @extract ep: λ
    - m*q1 + 1/(δqh+λ + m*(qh0 - qh1)) + (m*(qh0 + Power(ρh,2)))/Power(δqh+λ + m*(qh0 - qh1),2)
end
function fq1(op, ep)
    @extract op: qh0 qh1 δqh ρh m q1
    @extract ep: λ
    -(1/((δqh+λ)*m) - (qh0 + Power(ρh,2))/Power(δqh+λ + (qh0 - qh1)*m,2) -
      1/(m*(δqh+λ + (qh0 - qh1)*m)))
end
function fρ(op, ep)
    @extract op: qh0 qh1 δqh ρh m q1
    @extract ep: λ
    ρh/(δqh+λ + m*(qh0 - qh1))
end

function iρh(op, ep)
    @extract op: qh0 qh1 δqh m q1 ρ
    @extract ep: λ
    (true, ρ*(δqh+λ + m*(qh0 - qh1)))
end
function iδqh_fun(δq, op, ep)
    @extract op: qh0 qh1 ρh m q1
    @extract ep: λ
    0.5*q1 + (1/((δqh+λ)*m) - 1/(m*(δqh+ λ + m*(qh0 - qh1))) -
    (qh0 + Power(ρh,2))/Power(δqh+λ + m*(qh0 - qh1),2))/2.
end

function iδqh(op, ep, δqh₀, atol=1e-10)
    ok, δqh, it, normf0 = findroot(δqh -> iδqh_fun(δqh, op, ep), δqh₀, NewtonMethod(atol=atol))
    ok || error("iδqh failed: iδqh=$(δqh), it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, δqh
end

function im(op, ep; atol=1e-6)
    x₀ = op.m
    ok, x, it, normf0 = findroot(x->(op.m=x; ∂m_ϕ(op, ep)), x₀, NewtonMethod(atol=atol))
    ok || @warn("findroot failed: x=$(x), it=$it, normf0=$normf0")
    op.m = x₀
    return ok, ok ? x : x₀
end

function fhats_slow(op, ep)
    qh0 = qh1 = δqh = ρh = 0
    @sync begin
        qh0 = @spawn fqh0(op, ep)
        qh1 = @spawn fqh1(op, ep)
        δqh = @spawn fδqh(op, ep)
        ρh = @spawn fρh(op, ep)
    end
    return fetch(qh0), fetch(qh1), (fetch(qh1)-op.qh1)*op.m + fetch(δqh), fetch(ρh)
end


###############################

function fix_inequalities!(op, ep)
    if op.q0 < op.ρ^2
        op.q0 = op.ρ^2 + 1e-8
    end
    if (op.δqh+ep.λ + op.m*(op.qh0 - op.qh1)) < 0
        op.δqh = -op.m*(op.qh0-op.qh1)-ep.λ + 1e-8
    end
    if op.q1<op.q0
        op.q0 = op.q1 - 1e-9
    end
end


function converge!(op::OrderParams, ep::ExtParams, pars::Params;
        fixρ=false, fixnorm=false, fixm=true, testρ=false, testρ0=false)
    @extract pars: maxiters verb ϵ ψ solvenl
    Δ = Inf
    ok = false
    δq = op.δq
    q1 = op.q1
    q0 = op.q0

    ρ = op.ρ

    iterations = 0
    for it = 1:maxiters
        fix_inequalities!(op, ep)

        ### tentativo di accelerare la convergenza con
        ### il metodo di anderson, a volte è però instabile già per m=1.
        if solvenl > 0 && it%solvenl == 0
            println("# Calling NLsolve ...")
            @assert !fixnorm #not implemented
            try
                @show op
                res_nlsolve = nlsolve(x->nlsolve_fun(x, ep, fixρ=fixρ), collect(op),
                        method=:anderson, m=1, beta=1-ψ,
                        inplace=false, ftol=ϵ, iterations=10,
                        show_trace=true,extended_trace=true)
                println()
                for (i,x) in enumerate(res_nlsolve.zero)
                    setfield!(op, i, x)
                end
                fix_inequalities!(op, ep)
                @show op
            catch
                println("... NLsolve failed")
            end
        end

        ok = oki = true
        verb > 1 && println("it=$it")

        qh0, qh1, δqh, ρh = fhats_slow(op, ep)



        @update  op.qh0    identity       Δ ψ verb  qh0
        @update  op.qh1    identity       Δ ψ verb  qh1
        if fixnorm
            @updateI op.δqh ok   iδqh     Δ ψ verb  op ep
            ok &= oki
        else
            @update  op.δqh    identity     Δ ψ verb  δqh
        end
        if fixρ
            @updateI op.ρh oki   iρh   Δ ψ verb  op ep
            ok &= oki
        else
            @update  op.ρh  identity    Δ ψ verb  ρh
        end

        Δ = 0.0

        @update op.q0   fq0       Δ ψ verb  op ep
        op.q0 > 0 || (op.q0 = q0)
        if !fixnorm
            @update op.q1   fq1     Δ ψ verb  op ep
        end
        op.q1 > 0 || (op.q1 = q1)
        @update op.δq   fδq       Δ ψ verb  op ep
        op.δq > 0 || (op.δq = δq)
        if !fixρ
            @update op.ρ   fρ     Δ ψ verb  op ep
        end

        if !fixm
            @updateI op.m oki   im    Δ ψ verb  op ep op.m
            ok &= oki
        end


        op.q0<op.q1 || (op.q0=op.q1-1e-9)
        verb > 1 && println(" Δ=$Δ\n")
        verb > 4 && it%5==0 && (println(ep);println(all_therm_func(op, ep));println(op))

        if testρ
            op.ρ < 0.2*ρ && (return false, 0)
        elseif testρ0
            op.ρ > 0.3 && (return false, 0)
        end

        @assert isfinite(Δ)
        ok &= Δ < ϵ
        if ok
            iterations = it
            break
        end
        # op.δq > 1e4 && break # entering SAT PHASE
    end

    ok, iterations
end

function nlsolve_fun(x, ep; verb=1, fixρ=false)
    op = OrderParams(x...)
    for n in fieldnames(op) |> reverse
        if fixρ && n ∈ [:ρ, :ρh]
            n == :ρ && continue
            # implicit eq. for ρh
            f = (op, ep) -> getfield(PhaseRetr, Symbol(:i,n))(op, ep)[2]
        else
            f = getfield(PhaseRetr, Symbol(:f,n))
        end
        setfield!(op, n, f(op, ep))
    end
    collect(op) .- x
end

function converge(;
        q0 = 0.9,q1=1,
        δq=0.5,
        qh0=0., qh1=0.1,δqh=0.6,
        ρ=0, ρh=0, m=1.,
        α=0.1, λ= 0.,
        ϵ=1e-4, maxiters=10000, verb=3, ψ=0.,
        fixρ=false, fixnorm=false, fixm=true, solvenl=0)

    op = OrderParams(ρ,q0,q1,δq,ρh,qh0,qh1,δqh,m)
    ep = ExtParams(α,λ, 0)
    pars = Params(ϵ, ψ, maxiters, verb, solvenl)
    converge!(op, ep, pars,
        fixρ=fixρ,fixnorm=fixnorm,fixm=fixm)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
        q0 = 0.9,q1=1.,
        δq=0.3188,
        qh0=0.36889,qh1=0.6889,δqh=0.36889, ρh=0.56421,
        ρ=0.384312, α=1, m=1.,λ=0,
        ϵ=1e-4, maxiters=10000,verb=3, ψ=0., solvenl=0,
        kws...)

    op = OrderParams(first(ρ), q0,q1,δq,ρh,qh0,qh1,δqh,first(m))
    ep = ExtParams(first(α), λ, 0)
    pars = Params(ϵ, ψ, maxiters, verb, solvenl)
    return span!(op, ep, pars; ρ=ρ,α=α, m=m,kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
        αs=1, ρs=1, ms=1, λs=1,
        resfile = "results.txt",
        fixρ=false, fixnorm=false, fixm=true,
        check_ρstab=false, tol=-1# remain on uninf minimum (ρ=0) and check its stability toward the true signal
        )

    !isfile(resfile) && open(resfile, "w") do f
        if check_ρstab == false
            allheadersshow(f, ExtParams, ThermFunc, OrderParams)
        else
            allheadersshow(f, ExtParams, ThermFunc, OrderParams, "isρstab")
        end
    end

    lastm = first(ms)
    check_ρstab && (@assert fixρ=true && ρ==0)
    results = []
    for m in ms, α in αs, ρ in ρs, λ in λs
        fixm && (op.m = m)
        fixρ && (op.ρ = ρ)
        op.m, lastm = op.m+abs(op.m-lastm), op.m
        ep.α = α;
        ep.λ = λ;
        println("# NEW ITER: α=$(ep.α)  ρ=$(op.ρ)  m=$(op.m)")

        if fixm
            ok = converge!(op, ep, pars, fixm=true, fixρ=fixρ,fixnorm=fixnorm)
        else
            ok = findSigma0!(op, ep, pars; tol=tol>0 ? tol : pars.ϵ,fixρ=fixρ,fixnorm=fixnorm)
        end

        tf = all_therm_func(op, ep)
        tf.Σ < -1e-5 && @warn("Sigma negative")
        push!(results, (ok, deepcopy(ep), deepcopy(tf), deepcopy(op)))
        if check_ρstab
            @assert op.ρ == 0
            oop, ppars = deepcopy(op), deepcopy(pars)
            oop.ρ = pars.ϵ
            ppars.ϵ = 1e-100
            ppars.maxiters = 3
            converge!(oop, ep, ppars, fixm=true, fixρ=false,fixnorm=fixnorm)
            isρstab = abs(oop.ρ) < pars.ϵ ? 1 : 0
        end
        lastm = op.m
        ok && open(resfile, "a") do rf
            if !check_ρstab
                println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
            else
                pars.verb > 0 && println("# isρstable = ",isρstab)
                println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op), " ", isρstab)
            end
        end
        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end


function span_right!(op::OrderParams, ep::ExtParams, pars::Params;
        αs=-1, λs=-1,
        resfile = "results1RSB.txt",
        line = 1,
        fixnorm=false, extrap=-1, targetΣ=0., maximum_m=2000)

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    results = []

    ep, op, tf = readparams(resfile, line+1)
    for α in αs, λ in λs
        α > -1 && (ep.α = α)
        λ > -1 && (ep.λ = λ)
        println("# NEW ITER: α=$(ep.α)  ρ=$(op.ρ)  λ=$(ep.λ)  m=$(op.m)")

        ok = findSigma0!(op, ep, pars; tol=pars.ϵ,fixρ=false,fixnorm=fixnorm, targetΣ=targetΣ, testρ=true, maximum_m=maximum_m)
        ok || (return results)
        tf = all_therm_func(op, ep)
        tf.Σ < -1e-5 && @warn("Sigma negative")
        push!(results, (ok, deepcopy(ep), deepcopy(tf), deepcopy(op)))
        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end



function span_left!(op::OrderParams, ep::ExtParams, pars::Params;
    resfile = "results.txt",
    resfile2 = "results2.txt",
    line = 1,
    fixρ=true, fixm=true, targetΣ=0., maximum=false, fixnorm=false)

    !isfile(resfile2) && open(resfile2, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end


    ep, op, tf = readparams(resfile, line+1)

    println("# NEW ITER: α=$(ep.α)  ρ=$(op.ρ)  λ=$(ep.λ)  m=$(op.m)")

    op.ρ = 1e-5
    ok, iterations = converge!(op, ep, pars; fixρ=false, testρ0=false, fixnorm=fixnorm, fixm=true)
    ok || return
    tf = all_therm_func(op,ep)
    ep1 = deepcopy(ep)
    op1 = deepcopy(op)
    tf1 = deepcopy(tf)



    ep.λ = 0.
    ok, iterations2 = converge!(op, ep, pars; fixρ=false, testρ=true, fixnorm=fixnorm, fixm=true)
    open(resfile2, "a") do rf
        println(rf, plainshow(ep1), " ", plainshow(tf1), " ", plainshow(op1), " ", iterations, " ", iterations2, " ", op.ρ)
    end

    return
end



function span_atfixedm!(op::OrderParams, ep::ExtParams, pars::Params;
    resfile = "results.txt",
    resfile2 = "results2.txt",
    λ = -1, m = 1,
    fixρ=true, fixm=true, targetΣ=0., maximum=false, fixnorm=false)

    !isfile(resfile2) && open(resfile2, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    lambda = λ

    name = "temp$(lambda).txt"
    run(pipeline(`awk -v var=$(lambda) '$2==var {print}' "$(resfile)"`, stdout = name))
    a = readdlm(name)
    ms = Float64[]
    for k = 1:size(a,1)
        push!(ms, a[k, end])
    end
    l_min = findmin(abs.(ms .- m))[2]
    ep = ExtParams(a[l_min,1:3]...)
    op = OrderParams(a[l_min,7:15]...)
    println("# NEW ITER: α=$(ep.α)  ρ=$(op.ρ)  λ=$(ep.λ)  m=$(op.m)")
    while op.m != m
        op.m += sign(m-op.m) * min(abs(m - op.m), 0.1*op.m)
        println("# m=$(op.m)")
        ok, _ = converge!(op,ep,pars; fixρ=true, fixm=true, fixnorm=false)
    end
    tf = all_therm_func(op, ep)
    open(resfile2, "a") do rf
        println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
    end
    ep_min = deepcopy(ep)
    op_min = deepcopy(op)

    for α = ep_min.α+0.025:0.025:3.
        ep.α = α
        println("# NEW ITER: α=$(ep.α)  ρ=$(op.ρ)  λ=$(ep.λ)  m=$(op.m)")
        ok = false
        ok, _ = converge!(op,ep,pars; fixρ=true, fixm=true, fixnorm=false)
        ok || break
        tf = all_therm_func(op, ep)
        open(resfile2, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
    end
    ep = deepcopy(ep_min)
    op = deepcopy(op_min)
    for α = ep_min.α-0.025:-0.025:1.1
        ep.α = α
        println("# NEW ITER: α=$(ep.α)  ρ=$(op.ρ)  λ=$(ep.λ)  m=$(op.m)")
        ok = false
        ok, _ = converge!(op,ep,pars; fixρ=true, fixm=true, fixnorm=false)
        ok || break
        tf = all_therm_func(op, ep)
        open(resfile2, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
    end

    run(`rm $(name)`)
    return
end



function findSigma0!(   op, ep, pars;
                        tol = 1e-4, dm = 1, smallsteps = false, maxstep= 10.5,
                        fixρ=true, fixnorm=false, targetΣ = 0., maximum_m=-1, testρ=false, testρ0=false
                        )
    mlist = Any[]
    Σlist = Any[]

    if maximum_m>0 && op.m > maximum_m
        return false
    end
    ###PRIMO TENTATIVO
    println("@@@ T 1 : m=$(op.m)")

    converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm, testρ=testρ, testρ0=testρ0)
    if testρ && op.ρ<1e-5
        return false
    elseif testρ0 && op.ρ>0.3
        return false
    end

    tf = all_therm_func(op, ep)
    println(tf)
    push!(mlist, op.m)
    push!(Σlist, tf.Σ-targetΣ)
    absSigma = abs(tf.Σ-targetΣ)

    println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")

    ###SECOND TENTATIVO
    if absSigma > tol
        maxstep = abs(op.m) * 0.1
        dd = abs(op.m * (tf.Σ-targetΣ) * dm) > maxstep ? maxstep*sign(op.m * (tf.Σ-targetΣ) * dm) : op.m * (tf.Σ-targetΣ) * dm
        op.m += dd
        if maximum_m>0 && op.m > maximum_m
            return false
        end
        println("@@@ T 2 : m=$(op.m)")

        converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm, testρ=testρ, testρ0=testρ0)
        if testρ && op.ρ<1e-5
            return false
        elseif testρ0 && op.ρ>0.3
            return false
        end
        tf = all_therm_func(op, ep)
        println(tf)
        push!(mlist, op.m)
        push!(Σlist, (tf.Σ-targetΣ))
        absSigma = abs((tf.Σ-targetΣ))
        println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")
    end

    ###ALTRI  TENTATIVI
    trial = 3
    while absSigma > tol
        s = 0
        if trial >= 3
            s = -(mlist[end]*Σlist[end-1] - mlist[end-1]*Σlist[end])/(Σlist[end]-Σlist[end-1])
        end
        maxstep = abs(op.m) * 0.1
        dd = sign(s-op.m) * min(abs(s-op.m), maxstep)
        op.m += dd

        if maximum_m > 0 && (op.m > maximum_m)
            return false
        end
        # if smallsteps && abs(s - op.m) >  op.m * abs((tf.Σ-targetΣ)) * dm
            # dd = min(op.m * abs((tf.Σ-targetΣ)) * dm, maxstep)
            # op.m += sign(s - op.m) * dd
        # else
            # op.m = s
        # end
        println("@@@ T $(trial) : m=$(op.m)")
        converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm, testρ=testρ, testρ0=testρ0)
        if testρ && op.ρ<1e-5
            return false
        elseif testρ0 && op.ρ>0.3
            return false
        end

        tf = all_therm_func(op, ep)
        println(tf)
        println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")
        push!(mlist, op.m)
        push!(Σlist, (tf.Σ-targetΣ))
        absSigma = abs((tf.Σ-targetΣ))
        trial += 1
    end

    return true
end


function findmaximumSigma!(   Sigma0file; firstline = 1, lastline = 10000,
                        dm = 1,
                        fixρ=true, fixnorm=false, δ=1e-3,
                        ϵ=1e-4, maxiters=10000,verb=3, ψ=0., solvenl=0,
                        resfile = "results_1RSBmaximum.txt"
                        )
    Sfile = readdlm(Sigma0file)
    lines = size(Sfile, 1)
    pars = Params(ϵ, ψ, maxiters, verb, solvenl)

    for l = firstline:min(lastline, lines)

        ep, op, tf = readparams(Sigma0file, l+1)

        mlist = Any[]
        dΣlist = Any[]

        ###PRIMO TENTATIVO


        converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm)
        tf = all_therm_func(op, ep)
        println(tf)
        Σ_high = tf.Σ
        m_high = op.m

        times = 0
        ###SECOND TENTATIVO
        dm = op.m/20
        m_low = 0.
        while dm > δ
            println("going down")
            println("dm = $dm")
            Σ_low = Σ_high
            m_low = m_high
            while op.m > dm
                op.m -= dm
                println("@@@ m=$(op.m), (Σ_high,m_high)=($(Σ_high),$(m_high))")
                converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm)
                tf = all_therm_func(op, ep)
                println(tf)
                if tf.Σ > Σ_low
                    times += 1
                    Σ_high = Σ_low
                    m_high = m_low
                    Σ_low = tf.Σ
                    m_low = op.m
                else
                    while times < 2
                        println("going up")
                        Σ_low = Σ_high
                        m_low = m_high
                        op.m = m_high + dm
                        println("@@@ m=$(op.m), (Σ_low,m_low)=($(Σ_high),$(m_high))")
                        converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm)
                        tf = all_therm_func(op, ep)
                        println(tf)
                        if tf.Σ > Σ_high
                            times += 1
                            Σ_low = Σ_high
                            m_low = m_high
                            Σ_high = tf.Σ
                            m_high = op.m
                        else
                            Σ_high = tf.Σ
                            m_high = op.m
                            break
                        end
                    end
                    Σ_low = tf.Σ
                    m_low = op.m
                    break
                end
            end
            dm /= 5
        end
        open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
    end

    return true
end


function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line - 1 #since readdlm discards the header
    ep = ExtParams(res[line,1:3]...)
    op = OrderParams(res[line,7:15]...)
    tf = ThermFunc(res[line,4:6]...)
    return ep, op, tf
end

end ## module
