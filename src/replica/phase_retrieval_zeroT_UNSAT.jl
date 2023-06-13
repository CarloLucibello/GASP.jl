module PhaseRetr

using LittleScienceTools.Roots
using FastGaussQuadrature
using QuadGK
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 15.0
const dx = 0.1

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-9, reltol=1e-9, maxevals=5*10^3)[1]

function deriv(f::Function, i::Integer, x...; δ::Float64 = 1e-8)
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


let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    global gw
    gw(n::Int) = get!(s, n) do
        (x,w) = gausshermite(n)
        return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
    end
end

function ∫DD(f; n=121)
    (xs, ws) = gw(n)
    s = 0.0
    for (x,w) in zip(xs, ws)
        y = f(x)
        s += w  * ifelse(isfinite(y), y, 0.0)
    end
    return s
end

############### PARAMS

mutable struct OrderParams
    q0::Float64
    q1::Float64 # eventually q1=1
    δq::Float64
    qh0::Float64
    qh1::Float64
    δqh::Float64
    ρh::Float64
    m::Float64 # parisi breaking parameter
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
    Σ::Float64
    E::Float64
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
Gi(q0,q1,δq,qh0,qh1,δqh,ρh,m,ρ) = (q1*δqh - 2*ρ*ρh - δq*qh1 + q0*qh0*m - q1*qh1*m)/2

∂m_Gi(q0,q1,δq,qh0,qh1,δqh,ρh,m,ρ) = (q0*qh0 - q1*qh1) / 2

#### ENTROPIC TERM ####

Gs(qh0,qh1,δqh,ρh,m) = 0.5*((Power(ρh,2) + qh0)/(δqh + (qh0 - qh1)*m) + Log(δqh)/m - Log(δqh + (qh0 - qh1)*m)/m)

∂m_Gs(qh0,qh1,δqh,ρh,m) =  (-((qh0 - qh1)/(m*(δqh + m*(qh0 - qh1)))) -
    ((qh0 - qh1)*(qh0 + Power(ρh,2)))/Power(δqh + m*(qh0 - qh1),2) -
    Log(δqh)/Power(m,2) + Log(δqh + m*(qh0 - qh1))/Power(m,2)) / 2

#### ENERGETIC TERM ####

fy(ρ, q0, u0, z0) =  u0 * √(1-ρ^2/q0) + z0 * ρ/√(q0)
fyp(q0, q1, δq, z0, z1, u) = u * √(δq) + z1 * √(q1-q0) + z0 * √(q0)

fargGe(y, yp, u) = 1/2 * u^2 + 1/2 * (y^2 - yp^2)^2

function fargGe_min(y, q0, q1, δq, z0, z1; argmin=false)
    ### findmin of 1/2 u^2 + 1/2 * (y - (u √δq + z1 √(1-q0) + z0 √q0)^2)^2
    a = 9 * √(δq) * (√(q1-q0) * z1 + √(q0) * z0)
    b = 2 * y^2 * δq - 1
    c = 6*(-a + sqrt(complex(a^2 - 6 * b^3)))
    c3 = c^(1/3)
    bc3 = b/c3

    u1 = 1/δq * real((-a/9 - bc3 - c3/6))
    u2 = 1/δq * real((-a/9 + (1+√complex(-3))/2 * bc3 + (1-√complex(-3))/2 * c3/6))
    u3 = 1/δq * real((-a/9 + (1-√complex(-3))/2 * bc3 + (1+√complex(-3))/2 * c3/6))

    roots = [u1,u2,u3]
    if argmin
        m, am_ind = findmin(map(u->fargGe(y, fyp(q0, q1, δq, z0, z1, u), u), roots))
        am = roots[am_ind]
        yp = fyp(q0, q1, δq, z0, z1, am)
        return am, m, yp
    end
    minimum(r->fargGe(y, fyp(q0, q1, δq, z0, z1, r), r), roots)
end

function Ge(q0, q1, δq, ρ, m)
    ∫DD(z0->begin
        ∫DD(u0->begin
            y = fy(ρ, q0, u0, z0)
            log(∫D(z1->begin
                exp(-m * fargGe_min(y, q0, q1, δq, z0, z1; argmin=false))
            end))
        end)
    end) / m
end

function ∂ρ_Ge_an(q0, q1, δq, ρ, m)
    ∫DD(z0->begin
        ∫DD(u0->begin
            y = fy(ρ, q0, u0, z0)
            num = ∫D(z1->begin
                u, f, yp = fargGe_min(y, q0, q1, δq, z0, z1; argmin=true)
                - exp(-m * f) * u/√(δq) * (-u0/(√(1-ρ^2/q0))*(ρ/q0) + z0/√(q0))*y/yp
            end)
            den = ∫D(z1->begin
                exp(-m * fargGe_min(y, q0, q1, δq, z0, z1; argmin=false))
            end)
            num / den
        end)
    end)
end

function ∂q0_Ge_an(q0, q1, δq, ρ, m)
    ∫DD(z0->begin
        ∫DD(u0->begin
            y = fy(ρ, q0, u0, z0)
            num = ∫D(z1->begin
                u, f, yp = fargGe_min(y, q0, q1, δq, z0, z1; argmin=true)
                - exp(-m * f) * u/√(δq) * ((u0/(2*√(1-ρ^2/q0))*(ρ/q0)^2 - z0*ρ/(2*(q0)^(3/2)))*y/yp - (-z1/(2*√(q1-q0)) + z0/(2*√(q0))))
            end)
            den = ∫D(z1->begin
                exp(-m * fargGe_min(y, q0, q1, δq, z0, z1; argmin=false))
            end)
            num / den
        end)
    end)
end

function ∂q1_Ge_an(q0, q1, δq, ρ, m)
    ∫DD(z0->begin
        ∫DD(u0->begin
            y = fy(ρ, q0, u0, z0)
            num = ∫D(z1->begin
                u, f, yp = fargGe_min(y, q0, q1, δq, z0, z1; argmin=true)
                exp(-m * f) * u/√(δq) * (z1/(2*√(q1-q0)))
            end)
            den = ∫D(z1->begin
                exp(-m * fargGe_min(y, q0, q1, δq, z0, z1; argmin=false))
            end)
            num / den
        end)
    end)
end

function ∂δq_Ge_an(q0, q1, δq, ρ, m)
    ∫DD(z0->begin
        ∫DD(u0->begin
            y = fy(ρ, q0, u0, z0)
            num = ∫D(z1->begin
                u, f, yp = fargGe_min(y, q0, q1, δq, z0, z1; argmin=true)
                exp(-m * f) * u^2 / (2 * δq)
            end)
            den = ∫D(z1->begin
                exp(-m * fargGe_min(y, q0, q1, δq, z0, z1; argmin=false))
            end)
            num / den
        end)
    end)
end

function ∂m_Ge_an(q0, q1, δq, ρ, m)
    @sync begin
        dm = @spawn ∫DD(z0->begin
            ∫DD(u0->begin
                y = fy(ρ, q0, u0, z0)
                num = ∫D(z1->begin
                    f = fargGe_min(y, q0, q1, δq, z0, z1; argmin=false)
                    - exp(-m * f) * f
                end)
                den = ∫D(z1->begin
                    exp(-m * fargGe_min(y, q0, q1, δq, z0, z1; argmin=false))
                end)
                num / den
            end)
        end)/m
        ge = @spawn -Ge(q0, q1, δq, ρ, m) / m
    end
    return fetch(dm) + fetch(ge)
end

####

function ∂q0_Ge(q0, q1, δq, ρ, m)
    try
        return ∂q0_Ge_an(q0, q1, δq, ρ, m)
    catch
        return deriv(Ge, 1, q0, q1, δq, ρ, m)
    end
end
function ∂q1_Ge(q0, q1, δq, ρ, m)
    try
        return ∂q1_Ge_an(q0, q1, δq, ρ, m)
    catch
        return deriv(Ge, 2, q0, q1, δq, ρ, m)
    end
end
function ∂δq_Ge(q0, q1, δq, ρ, m)
    try
        return ∂δq_Ge_an(q0, q1, δq, ρ, m)
    catch
        return deriv(Ge, 3, q0, q1, δq, ρ, m)
    end
end
function ∂ρ_Ge(q0, q1, δq, ρ, m)
    try
        return ∂ρ_Ge_an(q0, q1, δq, ρ, m)
    catch
        return deriv(Ge, 4, q0, q1, δq, ρ, m)
    end
end
function ∂m_Ge(q0, q1, δq, ρ, m)
    try
        return ∂m_Ge_an(q0, q1, δq, ρ, m)
    catch
        return deriv(Ge, 5, q0, q1, δq, ρ, m)
    end
end


############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 q1 δq qh0 qh1 δqh ρh m
    @extract ep: α ρ
    Gi(q0,q1,δq,qh0,qh1,δqh,ρh,m,ρ) + Gs(qh0,qh1,δqh,ρh,m) + α*Ge(q0,q1,δq,ρ,m)
end

## Thermodinamic functions
# The energy of the pure states selected by x
# E = -∂(m*ϕ)/∂m
# if working at fixed x. If x is optimized Σ=0 and
# E = -ϕ
# so this formula is valid both at fixed and at optimized x
function all_therm_func(op::OrderParams, ep::ExtParams)
    @extract op: m
    ϕ = free_entropy(op, ep)
    E = -ϕ - m*im_fun(op, ep, m)
    Σ = m*(ϕ + E)
    return ThermFunc(ϕ, Σ, E)
end

###########################

### hat
fqh0(op, ep) = -2/op.m * ep.α * ∂q0_Ge(op.q0, op.q1, op.δq, ep.ρ, op.m)

fqh1(op, ep) = 2ep.α * ∂δq_Ge(op.q0, op.q1, op.δq, ep.ρ, op.m)

fρh(op, ep) = ep.α * ∂ρ_Ge(op.q0, op.q1, op.δq, ep.ρ, op.m)
iρh(op, ep) = (true, ep.ρ*(op.δqh + op.m*(op.qh0 - op.qh1)))

fδqh(op, ep) = op.m * op.qh1 - 2ep.α * ∂q1_Ge(op.q0, op.q1, op.δq, ep.ρ, op.m)
function iδqh_fun(qh0, qh1, δqh, ρh, m)
    0.5 + (1/(δqh*m) - 1/(m*(δqh + m*(qh0 - qh1))) -
    (qh0 + Power(ρh,2))/Power(δqh + m*(qh0 - qh1),2))/2.
end
function iδqh(op, ep; atol=1e-12)
    δqh₀ = op.δqh
    ok, δqh, it, normf0 = findroot(δqh -> iδqh_fun(op.qh0, op.qh1, δqh, op.ρh, op.m), δqh₀, NewtonMethod(atol=atol))

    ok || @warn("iδqh failed: δqh=$(δqh), it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, ok ? δqh : δqh₀
end
function fhats_slow(op, ep)
    qh0 = qh1 = δqh = 0
    @sync begin
        qh0 = @spawn fqh0(op, ep)
        qh1 = @spawn fqh1(op, ep)
        δqh = @spawn fδqh(op, ep)
    end
    return fetch(qh0), fetch(qh1), fetch(δqh)
end

### non hat
fq0(op, ep) = (op.qh0 + op.ρh^2) / (op.δqh + op.m*(op.qh0 - op.qh1))^2
fq1(op, ep) = - (1/(op.δqh*op.m) - 1/(op.m*(op.δqh + op.m*(op.qh0 - op.qh1))) - (op.qh0 + Power(op.ρh,2))/Power(op.δqh + op.m*(op.qh0 - op.qh1),2))
fδq(op, ep) = - op.m * op.q1 + 1/(op.δqh + op.m*(op.qh0 - op.qh1)) + (op.m*(op.qh0 + Power(op.ρh,2)))/Power(op.δqh + op.m*(op.qh0 - op.qh1),2)
fρ(op, ep) = op.ρh/(op.δqh + op.m*(op.qh0 - op.qh1))

function im_fun(op::OrderParams, ep::ExtParams, m)
    @extract op: q0 q1 δq qh0 qh1 δqh ρh
    @extract ep: α ρ
    ∂m_Gi(q0,q1,δq,qh0,qh1,δqh,ρh,m,ρ) + ∂m_Gs(qh0,qh1,δqh,ρh,m) + α*∂m_Ge(q0,q1,δq,ρ,m)
end

function im(op::OrderParams, ep::ExtParams, m₀, atol=1e-7)
    ok, m, it, normf0 = findroot(m -> im_fun(op, ep, m), m₀, NewtonMethod(atol=atol))
    ok || @warn("im failed: m=$m, it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, m
end

###############################

function fix_inequalities_hat(m, ρh, qh0, qh1, δqh)
    ok = false
    t = 0
    while !ok
        t += 1
        ok = true
        if δqh < 0
            δqh = 1e-4 * rand()
            ok = false
        end
        if δqh + m * (qh0 - qh1) < 1e-12
            δqh += 1e-4 * rand()
            ok = false
        end
    end
    t > 1 && println("***fixed***")
    return m, ρh, qh0, qh1, δqh
end

function fix_inequalities_nonhat(ρ, q0, q1, δq)
    ok = false
    t = 0
    while !ok
        ok = true
        t += 1
        if q0 < 0
            q0 += rand() * 1e-1
            ok = false
        end
        if q1 < 0
            q1 = q0 + rand()*1e-1
            ok=false
        end
        if q0 > q1
            q0 = q1 - rand() * 1e-1
            ok = false
        end
        if ρ > 1
            ρ = 1 - rand() * 1e-1
            ok = false
        end
        if ρ < 0
            ρ = rand() * 1e-1
            ok = false
        end
        if δq < 0
            δq = rand()
            ok = false
        end
        if ρ^2 / q0 > 1
            q0 += rand() * 1e-1
            ok = false
        end
    end
    t > 1 && println("***fixed***")
    return ρ, q0, q1, δq
end

###############################

# (OrderParams(q0=0.737905016841477,δq=5.013732664472831,qh0=0.003517416103730596,qh1=0.014003625691240064,δqh=0.20002992988871415,ρh=0.15163497624096378,m=1.0), ExtParams(α=1.6,ρ=0.8), Params(ϵ=0.0001,ψ=0.0,maxiters=200,verb=2), ThermFunc(ϕ=-0.004494791411858379,Σ=0.00014722936697765619,E=0.004642020778836035))

function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixm = true, fixρ = true, fixnorm = false)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    fixnorm && (op.q1 = 1)

    op.m, op.ρh, op.qh0, op.qh1, op.δqh = fix_inequalities_hat(op.m, op.ρh, op.qh0, op.qh1, op.δqh)
    ep.ρ, op.q0, op.q1, op.δq = fix_inequalities_nonhat(ep.ρ, op.q0, op.q1, op.δq)


    it = 0
    for it = 1:maxiters
        @time begin
            Δ = 0.0
            verb > 1 && println("it=$it")

            ### hats

            # @update  op.qh0    fqh0       Δ ψ verb  op ep
            # @update  op.qh1    fqh1       Δ ψ verb  op ep
            qh0, qh1, δqh = fhats_slow(op, ep)
            @update  op.qh0    identity       Δ ψ verb  qh0
            @update  op.qh1    identity       Δ ψ verb  qh1
            if fixnorm
                @updateI op.δqh ok   iδqh     Δ ψ verb  op ep
            else
                @update  op.δqh    identity     Δ ψ verb  δqh
            end
            if fixρ
                @updateI op.ρh ok   iρh   Δ ψ verb  op ep
            else
                @update  op.ρh  fρh       Δ ψ verb  op ep
            end
            op.m, op.ρh, op.qh0, op.qh1, op.δqh = fix_inequalities_hat(op.m, op.ρh, op.qh0, op.qh1, op.δqh)

            ### non hats

            @update op.q0   fq0       Δ ψ verb  op ep
            if !fixnorm
                @update op.q1   fq1       Δ ψ verb  op ep
            end
            @update op.δq   fδq       Δ ψ verb  op ep
            if !fixρ
                @update ep.ρ    fρ        Δ ψ verb  op ep
            end
            ep.ρ, op.q0, op.q1, op.δq = fix_inequalities_nonhat(ep.ρ, op.q0, op.q1, op.δq)

            ### m
            if !fixm
                @updateI op.m ok   im    Δ ψ verb  op ep op.m
            end
            # if it % 10 == 0
            #     println("  Σ = ",  -op.m^2 * im_fun(op, ep, op.m))
            # end
        end
        verb > 1 && println(" Δ=$Δ\n")

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end

end

# (OrderParams(q0=0.8369827702865094,δq=1.5456511977896277,qh0=0.009814760577267385,qh1=0.03337951886867011,δqh=0.647381861562955,ρh=0.1785729698541647,m=18.0), ExtParams(α=1.6,ρ=0.8), Params(ϵ=0.0001,ψ=0.0,maxiters=200,verb=2), ThermFunc(ϕ=-0.007500471295996312,Σ=-0.0009538678949934584,E=0.0074474786351633425))

function converge(;
    qh0 = 0.0007467601713782197,
  qh1 = 0.000955817614294672,
  δqh = 1.4775921697336318,
  ρh = 1.4627955513494466,
  q0 = 0.9804420458094858,
  q1 = 0.9805378010644004,
  δq = 0.6767767320939929,
    m=0.1,α=1.1,ρ=0.99,
    ϵ=1e-6, maxiters=100000, verb=2, ψ=0.,
    fixm = true, fixρ=true, fixnorm=false, resfile=nothing
    )

    op = OrderParams(q0,q1,δq,qh0,qh1,δqh,ρh,m)
    ep = ExtParams(α, ρ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixm=fixm, fixρ=fixρ, fixnorm=fixnorm)
    tf = all_therm_func(op, ep)
    println(tf)

    if resfile != nothing #&& tf.Σ > 0
        open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
    end

    return op, ep, pars, tf
end


function initialize_op(;qh0 = 0.002395415532596701,qh1 = 0.00960916849628631,
                        δqh = 0.16473126181432596,ρh = 0.12601400708050908,
                        q0 = 0.7365435338921105, q1 = 1., δq = 6.085044137391593,m=1.0,
                        α=1.2, ρ=0.2,
                        resfile=nothing, targetΣ=0.)

    op = 0
    if resfile == nothing
        op = OrderParams(q0,δq,qh0,qh1,δqh,ρh,m)
    else
        a = readdlm(resfile)
        a1 = map(i->a[i,1],1:size(a,1))
        a2 = map(i->a[i,2],1:size(a,1))
        a3 = map(i->a[i,4],1:size(a,1))
        l = a[findmin(map(i->abs(α-a1[i]) + abs(ρ-a2[i]) + abs(a3[i]-targetΣ), 1:length(a1)))[2],:]
        op = OrderParams(l[6:end]...)
    end
    return op
end

function spanSigma0(; lstα = [0.6], lstρ = [0.6], op = nothing,
                ϵ = 1e-4, ψ = 0., maxiters = 500, verb = 4,
                tol = 1e-4, dm = 10, smallsteps = true, maxstep=0.5,
                resfile = nothing, targetΣ = 0.
                )

    default_resfile = "results_1RSB_UNSAT_unconstrained.txt"
    resfile == nothing && (resfile = default_resfile)

    pars = Params(ϵ, ψ, maxiters, verb)
    results = Any[]

    ep = ExtParams(lstα[1], lstρ[1])

    for α in lstα, ρ in lstρ
        ep.α = α
        ep.ρ = ρ
        op = initialize_op(α=α, ρ=ρ, resfile=resfile, targetΣ=targetΣ)

        println()
        println("NEW POINT")
        println("       α=$(ep.α) ρ=$(ep.ρ)")
        println()

        tf = findSigma0!(op, ep, pars; tol=tol, dm=dm, smallsteps=smallsteps, maxstep=maxstep, targetΣ=targetΣ)

        push!(results, (deepcopy(ep), deepcopy(tf), deepcopy(op)))

        open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
    end
    return results, op
end

function findSigma0!(   op, ep, pars;
                        tol = 1e-4, dm = 10, smallsteps = true, maxstep= 0.5,
                        fixρ=true, fixnorm=false, targetΣ = 0.
                        )
    mlist = Any[]
    Σlist = Any[]

    ###PRIMO TENTATIVO
    println("@@@ T 1 : m=$(op.m)")

    converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm)
    tf = all_therm_func(op, ep)
    println(tf)
    push!(mlist, op.m)
    push!(Σlist, tf.Σ-targetΣ)
    absSigma = abs(tf.Σ-targetΣ)

    println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")

    ###SECOND TENTATIVO
    if absSigma > tol
        dd = abs(op.m * (tf.Σ-targetΣ) * dm) > maxstep ? maxstep*sign(op.m * (tf.Σ-targetΣ) * dm) : op.m * (tf.Σ-targetΣ) * dm
        op.m += dd
        println("@@@ T 2 : m=$(op.m)")

        converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm)
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
        if smallsteps && abs(s - op.m) >  op.m * abs((tf.Σ-targetΣ)) * dm
            dd = min(op.m * abs((tf.Σ-targetΣ)) * dm, maxstep)
            op.m += sign(s - op.m) * dd
        else
            op.m = s
        end
        println("@@@ T $(trial) : m=$(op.m)")
        converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm)

        tf = all_therm_func(op, ep)
        println(tf)
        println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")
        push!(mlist, op.m)
        push!(Σlist, (tf.Σ-targetΣ))
        absSigma = abs((tf.Σ-targetΣ))
        trial += 1
    end

    return tf
end
#
# function spanSigma0_optρ(; lstα = [0.6], initρ=0., op = nothing,
#                 ϵ = 1e-4, ψ = 0., maxiters = 500, verb = 4,
#                 tol = 1e-4, dm = 10, smallsteps = true, maxstep=0.5,
#                 resfile = nothing
#                 )
#
#     default_resfile = "results_phase_retrieval_zeroT_UNSAT.txt"
#     resfile == nothing && (resfile = default_resfile)
#
#     pars = Params(ϵ, ψ, maxiters, verb)
#     results = Any[]
#
#     ep = ExtParams(lstα[1], initρ)
#     op = initialize_op(α=lstα[1], ρ=initρ, resfile=resfile)
#
#     for α in lstα
#         ep.α = α
#
#         println()
#         println("NEW POINT")
#         println("α=$(ep.α)")
#         println()
#
#         tf = findSigma0!(op, ep, pars; tol=tol, dm=dm, smallsteps=smallsteps, maxstep=maxstep, fixρ=false)
#
#         push!(results, (deepcopy(ep), deepcopy(tf), deepcopy(op)))
#
#         open(resfile, "a") do rf
#             println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
#         end
#     end
#     return results, op
# end



end ## module
