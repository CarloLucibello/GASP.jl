module PhaseRetr

using LittleScienceTools.Roots
using QuadGK
using FastGaussQuadrature


include("../common.jl")


###### INTEGRATION  ######
const ∞ = 15.0
const dx = .25

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-6, reltol=1e-6, maxevals=2*10^3)[1]

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



function ∫DD(f; n=22)
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

Gi(m, ρ, q0, q1, ρh, qh0, qh1, qh2, β) = -0.5 * (2 * ρ * ρh + qh2 + (m - 1) * q1 * qh1 - m * q0 * qh0 - log(β/(2π)))

∂m_Gi(q0, q1, qh0, qh1) = -0.5 * (q1 * qh1 - q0 * qh0)

#### ENTROPIC TERM ####
Gs(m, ρh, qh0, qh1, qh2) = 0.5 * ((qh0 + ρh^2)/(qh1 - qh2 + m * (qh0 - qh1)) + (1-m) /m * log(qh1 - qh2) - 1/m * log(qh1 - qh2 + m * (qh0 - qh1)))

∂m_Gs(m, ρh, qh0, qh1, qh2) = 0.5 * ((qh1 - qh0)/(m * (qh0 - qh1) + qh1 - qh2) * ((qh0 + ρh^2)/(m * (qh0 - qh1) + qh1 - qh2) + 1/m) + (log(m * (qh0 - qh1) + qh1 - qh2) - log(-qh2 + qh1)) / m^2 )
∂ρh_Gs(m, ρh, qh0, qh1, qh2) = ρh / (m * (qh0 - qh1) + qh1 - qh2)
∂qh0_Gs(m, ρh, qh0, qh1, qh2) = -(m * (ρh^2 + qh0))/(2 * (m * (qh0 - qh1) + qh1 - qh2)^2)
∂qh1_Gs(m, ρh, qh0, qh1, qh2) = 0.5 * ((1 - m)/(m * (qh1 - qh2)) - (1 - m) /(m * (qh0 - qh1) + qh1 - qh2) * ((ρh^2 + qh0)/(m * (qh0 - qh1) + qh1 - qh2) + 1/m))

∂qh2_Gs(m, ρh, qh0, qh1, qh2) = 0.5 * (-(1 - m)/(m * (qh1 - qh2)) + (ρh^2 + qh0)/(m * (qh0 - qh1) + qh1 - qh2)^2 + 1/(m * (m * (qh0 - qh1) + qh1 - qh2)))

#### ENERGETIC TERM ####

fy(ρ, q0, u0, z0) = (sqrt(1 - ρ^2 / q0) * u0 + ρ / sqrt(q0) * z0)^2
fargGe(y, q0, q1, z0, z1, u, β) = 1/2 * u^2 + β * 1/2 * (y - (u * √(1-q1) + z1 * √(q1-q0) + z0 * √(q0))^2)^2
exp_argGe(y, q0, q1, z0, z1, u) = 1/2 * (y - (u * √(1-q1) + z1 * √(q1-q0) + z0 * √(q0))^2)^2

function fargGe_min(y, q0, q1, z0, z1, β)
    ### findmin of 1/2 u^2 + β * 1/2 * (y - (u √δq + z1 √(q1-q0) + z0 √q0)^2)^2
    a = 9 * β^2 * √(1-q1) * (z1 * √(q1-q0) + z0 * √(q0))
    b = β * (2*y*β*(1-q1) - 1)
    c = 6*(-a + sqrt(complex(a^2 - 6*b^3)))

    r1 = 1/(β*(1-q1)) * real((-a/(9*β) -b/c^(1/3) - c^(1/3)/6))
    r2 = 1/(β*(1-q1)) * real((-a/(9*β) + (1+√complex(-3))/2 * b/c^(1/3) + (1-√complex(-3))/2 * c^(1/3)/6))
    r3 = 1/(β*(1-q1)) * real((-a/(9*β) + (1-√complex(-3))/2 * b/c^(1/3) + (1+√complex(-3))/2 * c^(1/3)/6))
    [r1, r2, r3][findmin(map(r->fargGe(y, q0, q1, z0, z1, r, β), [r1, r2, r3]))[2]]
end

function inner_integral(y, q0, q1, z0, z1, β; spacings=∞*(-1:0.5:1))
    # min_phase = fargGe_min(y, q0, q1, z0, z1, β)
    #  ∫D_centered(u->begin
    #  exp(- β * exp_argGe(y, q0, q1, z0, z1, u))
    # end, min_phase, β, int = spacings)
     ∫D(u->begin
        exp(- β * exp_argGe(y, q0, q1, z0, z1, u))
    end)
end


function Ge(q0, q1, ρ, m, β)
    spacings = uneven_spacings(β=3., intervals=5, ∞=15.)
    ∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            log(∫D(z1->begin
            (inner_integral(y, q0, q1, z0, z1, β, spacings=spacings))^m
        end))
    end)
end) / m
end

function ∂q1_Ge_an(q0, q1, ρ, m, β)
    spacings = uneven_spacings(β=3., intervals=5, ∞=15.)
    ∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            num = ∫D(z1->begin
            m * (inner_integral(y, q0, q1, z0, z1, β, spacings=spacings))^(m-1) * ∫D(u->begin
            exp(- β * exp_argGe(y, q0, q1, z0, z1, u)) * β * (y - (u * √(1-q1) + z1 * √(q1-q0) + z0 * √(q0))^2) * 2 * (u * √(1-q1) + z1 * √(q1-q0) + z0 * √(q0)) * (-1/(2*√(1-q1)) * u + 1/(2*√(q1-q0)) * z1) end)
        end)
            den = ∫D(z1->begin
            (inner_integral(y, q0, q1, z0, z1, β, spacings=spacings))^m
        end)
            num / den
        end)
    end) / m
end

function ∂q0_Ge_an(q0, q1, ρ, m, β)
    spacings = uneven_spacings(β=3., intervals=5, ∞=15.)
    ∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            num = ∫D(z1->begin
            m * (inner_integral(y, q0, q1, z0, z1, β, spacings=spacings))^(m-1) * ∫D(u->begin
            exp(- β * exp_argGe(y, q0, q1, z0, z1, u)) * β * (y - (u * √(1-q1) + z1 * √(q1-q0) + z0 * √(q0))^2) * 2 * ((u * √(1-q1) + z1 * √(q1-q0) + z0 * √(q0)) * (- 1/(2*√(q1-q0)) * z1 + 1/(2*√(q0))*z0) - (√(1-ρ^2/q0) * u0 + ρ/√(q0) * z0)*(1/(2*√(1-ρ^2/q0))*ρ^2/q0^2*u0 - ρ/(2*q0^(3/2))*z0)) end)
        end)
            den = ∫D(z1->begin
            (inner_integral(y, q0, q1, z0, z1, β, spacings=spacings))^m
        end)
            num / den
        end) / m
    end)
end

function ∂m_Ge(q0, q1, ρ, m, β)
    spacings = uneven_spacings(β=3., intervals=5, ∞=15.)
    ∫DD(u0->begin
        ∫DD(z0->begin
            y = fy(ρ, q0, u0, z0)
            num = ∫D(z1->begin
            integr = inner_integral(y, q0, q1, z0, z1, β, spacings=spacings)
            integr^m * log(integr)
        end)
            den = ∫D(z1->begin
            (inner_integral(y, q0, q1, z0, z1, β, spacings=spacings))^m
        end)
            num / den
        end)
    end) / m - 1 / m * Ge(q0, q1, ρ, m, β)
end

function ∂q0_Ge(q0, q1, ρ, m, β)
    try
        ∂q0_Ge_an(q0, q1, ρ, m, β)
    catch
        deriv(Ge, 1, q0, q1, ρ, m, β)
    end
end
function  ∂q1_Ge(q0, q1, ρ, m, β)
    try
        ∂q1_Ge_an(q0, q1, ρ, m, β)
    catch
        deriv(Ge, 2, q0, q1, ρ, m, β)
    end
end
∂ρ_Ge(q0, q1, ρ, m, β) = deriv(Ge, 3, q0, q1, ρ, m, β)
# ∂m_Ge(q0, q1, ρ, m, β) = deriv(Ge, 4, q0, q1, ρ, m, β)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 q1 qh0 qh1 qh2 ρh m
    @extract ep: α β ρ
    Gi(m, ρ, q0, q1, ρh, qh0, qh1, qh2, β) + Gs(m, ρh, qh0, qh1, qh2) + α*Ge(q0, q1, ρ, m, β)
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
    Σ = - m^2 * im_fun(op, ep, m)
    F = (m*ϕ - Σ) /(-β*m)
    return ThermFunc(ϕ, Σ, F)
end

###########################
#### SADDLE POINT EQUATIONS ####
#
# q00 = 1
# q2 = 1
# q0 = -2/m * ∂q̂0_Gs
# q1 = 2/(m-1) * ∂q̂1_Gs
# q̂00 = -1
#
# q̂0 = -2α/m * ∂q0_Ge
# q̂1 = 2α/(m-1) * ∂q1_Ge
# 0 = -1/2 +  ∂q̂2_Gs --->>> q̂2
# 0 = ∂p̂_Gs - p --->>> p  or  p̂

fqh0(q0, q1, ρ, m, α, β) = -2/m * α * ∂q0_Ge(q0, q1, ρ, m, β)
fqh1(q0, q1, ρ, m, α, β) = 2/(m-1) * α * ∂q1_Ge(q0, q1, ρ, m, β)
fρh(q0, q1, ρ, m, α, β) = α * ∂ρ_Ge(q0, q1, ρ, m, β)

fq0(m, ρh, qh0, qh1, qh2) = -2/m * ∂qh0_Gs(m, ρh, qh0, qh1, qh2)
fq1(m, ρh, qh0, qh1, qh2) = 2/(m-1) * ∂qh1_Gs(m, ρh, qh0, qh1, qh2)
fρ(m, ρh, qh0, qh1, qh2) = ρh / (m * (qh0 - qh1) + qh1 - qh2)

iρh(ρ, qh0, qh1, qh2, m) = (true, ρ*(qh1 - qh2 + m*(qh0 - qh1)))


function iqh2_fun(qh0, qh1, qh2, ρh, m)
    -0.5 + ∂qh2_Gs(m, ρh, qh0, qh1, qh2)
end

function iqh2(qh0, qh1, qh2₀, ρh, m, atol=1e-12)
    ok, qh2, it, normf0 = findroot(qh2 -> iqh2_fun(qh0, qh1, qh2, ρh, m), qh2₀, NewtonMethod(atol=atol))
    #ok, M, it, normf0 = findzero_interp(M->∂_Ge(5, Q, q0, q1, δq, M, x, K, avgξ, varξ, f′), M0, dx=0.1)

    ok || @warn("iqh2 failed: iqh2=$(qh2), it=$it, normf0=$normf0")
    # ok = normf0 < 1e-5
    return ok, ok ? qh2 : qh2₀
end

function im_fun(op::OrderParams, ep::ExtParams, m)
    @extract op: q0 q1 qh0 qh1 qh2 ρh m
    @extract ep: α β ρ
    ∂m_Gi(q0, q1, qh0, qh1) + ∂m_Gs(m, ρh, qh0, qh1, qh2) + α*∂m_Ge(q0, q1, ρ, m, β)
end

function im(op::OrderParams, ep::ExtParams, m₀, atol=1e-7)
    ok, m, it, normf0 = findroot(m -> im_fun(op, ep, m), m₀, NewtonMethod(atol=atol))
    ok || @warn("im failed: m=$m, it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, m
end

###############################

function fix_inequalities_hat(m, ρh, qh0, qh1, qh2)
    ok = false
    t = 0
    while !ok
        t += 1
        ok = true
        if qh1 < qh2
            qh1 += 1e-4 * rand()
            qh2 -= 1e-4 * rand()
            ok = false
        end
        if qh1 - qh2 + m * (qh0 - qh1) < 0
            qh0 += 1e-4 * rand()
            qh2 += 1e-4 * rand()
            ok = false
        end
    end
    t > 1 && println("***fixed***")
    return t, m, ρh, qh0, qh1, qh2
end

function fix_inequalities_nonhat(ρ, q0, q1)
    ok = false
    t = 0
    while !ok
        ok = true
        t += 1
        if q0 < 0
            q0 = rand() * 1e-4
            ok = false
        end
        if q0 > q1
            q0 -= rand() * 1e-4
            q1 += rand() * 1e-4
            ok = false
        end
        if q0 > 1
            q0 = q1 - rand() * 1e-4
            ok = false
        end
        if q1 < 0
            q1 = q0 + rand() * 1e-4
            ok = false
        end
        if q1 > 1
            q1 = 1 - rand() * 1e-4
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
    return t, ρ, q0, q1
end

###############################

function fallhats(q0, q1, ρ, m, α, β)
    @sync begin
        qh0 = @spawn fqh0(q0, q1, ρ, m, α, β)
        qh1 = @spawn fqh1(q0, q1, ρ, m, α, β)
    end
    fetch(qh0), fetch(qh1)
end

function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixm = false, fixρ = true)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    t, op.m, op.ρh, op.qh0, op.qh1, op.qh2 = fix_inequalities_hat(op.m, op.ρh, op.qh0, op.qh1, op.qh2)
    t, ep.ρ, op.q0, op.q1 = fix_inequalities_nonhat(ep.ρ, op.q0, op.q1)


    it = 0
    for it = 1:maxiters
        @time begin
            Δ = 0.0
            verb > 1 && println("it=$it")
            qh0 = 0
            qh1 = 0


            qh0, qh1 = fallhats(op.q0, op.q1, ep.ρ, op.m, ep.α, ep.β)
            @update  op.qh0      identity     Δ ψ verb  qh0
            @update  op.qh1      identity     Δ ψ verb  qh1


            # @update  op.qh0      fqh0       Δ ψ verb  op.q0 op.q1 ep.ρ op.m ep.α ep.β
            # @update  op.qh1      fqh1       Δ ψ verb  op.q0 op.q1 ep.ρ op.m ep.α ep.β

            @updateI op.qh2   ok iqh2       Δ ψ verb  op.qh0 op.qh1 op.qh2 op.ρh op.m  ϵ
            if fixρ
                @updateI op.ρh  ok iρh      Δ ψ verb  ep.ρ op.qh0 op.qh1 op.qh2 op.m
            else
                @update  op.ρh   fρh        Δ ψ verb  op.q0 op.q1 ep.ρ op.m ep.α
            end
            t, op.m, op.ρh, op.qh0, op.qh1, op.qh2 = fix_inequalities_hat(op.m, op.ρh, op.qh0, op.qh1, op.qh2)
            t > 1 && (ok = false)

            @update op.q0        fq0        Δ ψ verb  op.m op.ρh op.qh0 op.qh1 op.qh2
            # op.q1 = op.q0
            @update op.q1        fq1        Δ ψ verb  op.m op.ρh op.qh0 op.qh1 op.qh2
            if !fixρ
                @update ep.ρ     fρ         Δ ψ verb  op.m op.ρh op.qh0 op.qh1 op.qh2
            end
            t, ep.ρ, op.q0, op.q1 = fix_inequalities_nonhat(ep.ρ, op.q0, op.q1)
            t > 1 && (ok = false)


            if !fixm
                @updateI op.m ok   im    Δ ψ verb  op ep op.m  ϵ/10
            end

        end

        verb > 1 && println(" Δ=$Δ\n")

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end

end

###########
### RS branch
# (OrderParams(q0=0.2585013158969293,q1=0.25916954573603695,qh0=0.015447930512553939,qh1=0.016663808284050147,qh2=-1.3331512181740632,ρh=0.6739954732280717,m=1.5), ExtParams(α=0.1,β=1.0,ρ=0.5), Params(ϵ=1.0e-5,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-0.6148824774726193,Σ=0.0017065657617532236,F=0.6160201879804547))

# (OrderParams(q0=0.2583423749174141,q1=0.25916793669608273,qh0=0.01515855083821793,qh1=0.01666134583045512,qh2=-1.3331719237163295,ρh=0.6740151554353514,m=1.2), ExtParams(α=0.1,β=1.0,ρ=0.5), Params(ϵ=1.0e-5,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-0.6148825511799105,Σ=0.0016172831463104479,F=0.6162302871351693))

# (OrderParams(q0=0.2580852005951037,q1=0.25916441502935084,qh0=0.014692786794131775,qh1=0.016656110879141383,qh2=-1.3331635115384637,ρh=0.6740263258960737,m=0.9), ExtParams(α=0.1,β=1.0,ρ=0.5), Params(ϵ=1.0e-5,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-0.6148826700809487,Σ=0.0015159726220512516,F=0.6165670841054501))

# (OrderParams(q0=0.25631525838745917,q1=0.2595266012698649,qh0=0.011428272091486448,qh1=0.017262763024869923,qh2=-1.3332534439045691,ρh=0.6726328146188507,m=0.9), ExtParams(α=0.1,β=1.1,ρ=0.5), Params(ϵ=1.0e-5,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-0.5697811302135918,Σ=0.004508840884596837,F=0.5225372303806357))

#(OrderParams(q0=0.6413702755145392,q1=0.6420052091607151,qh0=0.01065934426870741,qh1=0.015605132845616229,qh2=-2.7775852559507723,ρh=2.230991032508385,m=0.9), ExtParams(α=0.1,β=1.25,ρ=0.8), Params(ϵ=1.0e-5,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-0.8563459236164065,Σ=0.002886561372373544,F=0.6876425712241239))

# (OrderParams(q0=0.6414786545000893,q1=0.6422620523190306,qh0=0.011510136053836627,qh1=0.017618864773331987,qh2=-2.7775792246124156,ρh=2.231759874809262,m=0.9), ExtParams(α=0.1,β=1.5,ρ=0.8), Params(ϵ=1.0e-5,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-0.7692589379038409,Σ=0.003566558826860058,F=0.5154811873631977))

# (OrderParams(q0=0.36788896544578803,q1=0.36976273064597975,qh0=0.019802897754535333,qh1=0.024513621811058473,qh2=-1.562236077702958,ρh=0.9506368211543472,m=0.5), ExtParams(α=0.1,β=3.0,ρ=0.6), Params(ϵ=1.0e-5,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-0.17154063585469806,Σ=0.0017037172541581075,F=0.05831602345433809))

# (OrderParams(q0=0.37821128230886725,q1=0.3802251127179035,qh0=0.04726221639004828,qh1=0.0524951887698744,qh2=-1.5609390455713994,ρh=0.9664883216624036,m=0.5), ExtParams(α=0.2,β=3.0,ρ=0.6), Params(ϵ=1.0e-5,ψ=0.3,maxiters=100000,verb=2), ThermFunc(ϕ=-0.25041231145150045,Σ=0.0018724003215760266,F=0.08471903736488418))

# (OrderParams(q0=0.41534497391902264,q1=0.41552194589308145,qh0=0.16195784306554906,qh1=0.16247571400326472,qh2=-1.5483919373746047,ρh=1.026362914646307,m=0.5), ExtParams(α=0.5,β=3.0,ρ=0.6), Params(ϵ=1.0e-5,ψ=0.3,maxiters=100000,verb=2), ThermFunc(ϕ=-0.48795545119166617,Σ=0.0001782664117681417,F=0.16277066133840082))

# (OrderParams(q0=0.47916297644745853,q1=0.4791753770847409,qh0=0.43929819949355764,qh1=0.4393438984042917,qh2=-1.4807178654975435,ρh=1.1520239309010882,m=0.5), ExtParams(α=1.2,β=1.5,ρ=0.6), Params(ϵ=1.0e-5,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-1.141887764690518,Σ=1.463777253377947e-5,F=0.7612780268237236))

# (OrderParams(q0=0.5285619422549518,q1=0.5330911309584035,qh0=0.761295628495323,qh1=0.7819094170485598,qh2=-1.359789453429267,ρh=1.2751295561317453,m=0.8), ExtParams(α=1.2,β=3.5,ρ=0.6), Params(ϵ=0.0001,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-1.0358744800704005,Σ=0.011310778861569537,F=0.30000370104210355))

# (OrderParams(q0=0.5534651591232422,q1=0.5582553123932319,qh0=0.9743510694841163,qh1=0.9986873187445587,qh2=-1.2649685455117379,ρh=1.3465156504248574,m=0.8), ExtParams(α=1.2,β=5.5,ρ=0.6), Params(ϵ=0.0001,ψ=0.1,maxiters=100000,verb=2), ThermFunc(ϕ=-0.9997859823519499,Σ=0.01308344957403007,F=0.18475278078536136))
###########

##########
### 1RSB branch
# (OrderParams(q0=0.3168569390936883,q1=0.5916699432016157,qh0=0.6443653950124584,qh1=1.6716218746568345,qh2=-0.7785060533128418,ρh=0.30511895539498385,m=0.9), ExtParams(α=1.2,β=10.5,ρ=0.2), Params(ϵ=0.0001,ψ=0.3,maxiters=100000,verb=2), ThermFunc(ϕ=-1.003290960742572,Σ=-0.0031790154986261554,F=0.09521511631425275))
# (OrderParams(q0=0.334097162294009,q1=0.6589817149028472,qh0=0.7334604654142794,qh1=2.2389243692122003,qh2=-0.695223432738749,ρh=0.31584671680348736,m=0.9), ExtParams(α=1.2,β=15.0,ρ=0.2), Params(ϵ=0.0001,ψ=0.3,maxiters=100000,verb=2), ThermFunc(ϕ=-0.9958376019927428,Σ=-0.005517938235570608,F=0.06598043730058503))
# (OrderParams(q0=0.313602440905425,q1=0.642151641376935,qh0=0.7097683906733747,qh1=2.1878192861514174,qh2=-0.6052654900120655,ρh=0.3221350548547566,m=0.8), ExtParams(α=1.2,β=15.0,ρ=0.2), Params(ϵ=0.0001,ψ=0.3,maxiters=100000,verb=2), ThermFunc(ϕ=-0.9963959900139608,Σ=-0.0024451166892149882,F=0.06622263961016281))
# (OrderParams(q0=0.2913218460665935,q1=0.613087489697848,qh0=0.6702884435396497,qh1=2.0276912922626558,qh2=-0.5555807176233086,ρh=0.3266219570553873,m=0.7), ExtParams(α=1.2,β=15.0,ρ=0.2), Params(ϵ=0.0001,ψ=0.3,maxiters=100000,verb=2), ThermFunc(ϕ=-0.9966168867792989,Σ=-0.0002214661546361049,F=0.06642003377055934))
######

#### Σ = 0
# (OrderParams(q0=0.2890970322394353,q1=0.6088479480576009,qh0=0.6653033814291617,qh1=2.0009873206807525,qh2=-0.5550448647667288,ρh=0.32685533979399173,m=0.6901), ExtParams(α=1.2,β=15.0,ρ=0.2), Params(ϵ=0.0001,ψ=0.0,maxiters=100000,verb=2), ThermFunc(ϕ=-0.9966250670593566,Σ=-5.5553996237665545e-5,F=0.06643630437921308))
# (OrderParams(q0=0.2917363045005529,q1=0.5990594266092554,qh0=0.6696783551789279,qh1=1.9195932347167075,qh2=-0.5739964727174345,ρh=0.3262046898130241,m=0.6901), ExtParams(α=1.2,β=15.0,ρ=0.2), Params(ϵ=0.0001,ψ=0.0,maxiters=100000,verb=2), ThermFunc(ϕ=-0.9966855253555506,Σ=-5.643786369230639e-5,F=0.06644024954684569))
# (OrderParams(q0=0.36429510658476527,q1=0.5383657173657839,qh0=0.8297888104985036,qh1=1.48542786302198,qh2=-0.6801041270648315,ρh=0.5217903126123897,m=0.6501), ExtParams(α=1.2,β=15.0,ρ=0.3), Params(ϵ=0.0001,ψ=0.0,maxiters=100000,verb=2), ThermFunc(ϕ=-0.9873572308230778,Σ=-7.693679766954433e-5,F=0.06581592564840419))
####

function converge(;q0=0.313602440905425,q1=0.642151641376935,qh0=0.7097683906733747,qh1=2.1878192861514174,qh2=-0.6052654900120655,ρh=0.3221350548547566,m=0.8,
        α=1.2,β=15.0,ρ=0.2,
        ϵ=1.0e-4,ψ=0.0,maxiters=100000,verb=2,
        fixm = false, fixρ=true
    )
    op = OrderParams(q0,q1,qh0,qh1,qh2,ρh,m)
    ep = ExtParams(α, β, ρ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixm=fixm, fixρ=fixρ)
    tf = all_therm_func(op, ep)
    println(tf)
    println()
    return op, ep, pars, tf
end


function initialize_op(  qh0 = 0.4452202195474727,
qh1 = 1.8911367908281913,
qh2 = -0.61331634019495,
ρh = 0.9401089264798257,
q0 = 0.3181599005710799,
q1 = 0.6006377372682961,m=0.2)
    op = OrderParams(q0, q1, qh0, qh1, qh2, ρh, m)
    return op
end

function spanSigma0(; lstα = [0.6], lstρ = [0.6], β=15., op = nothing,
                ϵ = 1e-5, ψ = 0.1, maxiters = 500, verb = 4,
                resfile = nothing, updatem = false,
                tol = 1e-4, dm = 10, smallsteps = true
                )

    default_resfile = "results_phase_retrieval_finiteT.txt"
    resfile == nothing && (resfile = default_resfile)

    pars = Params(ϵ, ψ, maxiters, verb)
    results = Any[]

    lockfile = "reslock.tmp"

    op == nothing && (op = initialize_op())
    ep = ExtParams(lstα[1], β, lstρ[1])
    println(op)

    for α in lstα, ρ in lstρ
        ep.α = α
        ep.ρ = ρ

        println()
        println("NEW POINT")
        println("       α=$(ep.α) ρ=$(ep.ρ)")
        println()

        if !updatem
            tf = findSigma0!(op, ep, pars; tol=tol, dm=dm, smallsteps=smallsteps)
        else
            converge!(op, ep, pars, updatem = true)
        end

        push!(results, (deepcopy(op), deepcopy(ep), deepcopy(tf)))


        open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
    end
    return results, op
end

function findSigma0!(   op, ep, pars;
                        tol = 1e-4, dm = 10, smallsteps = true
                        )

    mlist = Any[]
    Σlist = Any[]

    ###PRIMO TENTATIVO
    println("@@@ TRIAL 1 SIGMA0")

    converge!(op, ep, pars, fixm=true, fixρ=true)
    tf = all_therm_func(op, ep)
    println(tf)
    push!(mlist, op.m)
    push!(Σlist, tf.Σ)
    absSigma = abs(tf.Σ)

    println("\n @@@  m=$(op.m)  Σ=$(tf.Σ) \n")

    ###SECOND TENTATIVO
    if absSigma > tol
        println("@@@ TRIAL 2 SIGMA0")
        op.m += op.m * tf.Σ * dm
        println("@@@  m=$(op.m)")

        converge!(op, ep, pars, fixm=true, fixρ=true)
        tf = all_therm_func(op, ep)
        println(tf)
        push!(mlist, op.m)
        push!(Σlist, tf.Σ)
        absSigma = abs(tf.Σ)
        println("\n @@@  m=$(op.m) Σ=$(tf.Σ) \n")
    end

    ###ALTRI  TENTATIVI
    trial = 3
    while absSigma > tol
        println("TRIAL $(trial) SIGMA0")
        dummyfile = string("dummy.dat")
        rf = open(dummyfile, "w")
        m = max(1, length(mlist)-4)
        for i = m:length(mlist)
            println(rf, "$(mlist[i]) $(Σlist[i])")
        end
        close(rf)
        s = 0
        if trial >= 3 || trial == 4
            s = -(mlist[end]*Σlist[end-1] - mlist[end-1]*Σlist[end])/(Σlist[end]-Σlist[end-1])
        else
            s = run(`gnuplot -e "filename='$dummyfile'" interpolation.gpl `)
        end
        run(`rm $dummyfile`)
        if smallsteps && abs(s - op.m) >  op.m * abs(tf.Σ) * dm
            op.m += sign(s - op.m) * op.m * abs(tf.Σ) * dm
        else
            op.m = s
        end

        println("@@@  m=$(op.m)")
        converge!(op, ep, pars, fixm=true, fixρ=true)

        tf = all_therm_func(op, ep)
        println(tf)
        println("\n @@@  m=$(op.m) Σ=$(tf.Σ) \n")
        push!(mlist, op.m)
        push!(Σlist, tf.Σ)
        absSigma = abs(tf.Σ)
        trial += 1
    end

    println("\n\n@@@ SUCCESS SIGMA0\n\n")

    return tf
end



# function followSigma0( ;α = 0.5, D=0.2, y=5.,  avgξ=0., varξ=1., f′=0.5, K=0., ls=[-1,1],
#                     qt = 0.5, q0 = 0.51, q1 = 0.6, Qt = 1., Q = 1., St = 0.3, S = 0.9, Mt=0., M=0., avgWt = 0., avgW= 0.,
#                     q̂t = 0.5, q̂0 = 0.36, q̂1 = 0.45, Q̂t = 0.1, Q̂ = 0.14, Ŝt = 0.25, Ŝ = 0.3, D̂ = 0.1,
#                     ϵ = 1e-4, ∂ϵ = 1e-6, ψ = 0.0, maxiters = 1000, verb = 2,
#                     nint1 = 5, nint2 = 200, nint3 = 200,
#                     tol = 1e-4, xlist = 0.5, dy = 10, smallsteps = false, withq1 = true,
#                     resfile = nothing, extrapol = true, with_D = false)
#
#     default_resfile = "results_franz-parisi_followSigma0.generalized.txt"
#     resfile == nothing && (resfile = default_resfile)
#     lockfile = "reslock.tmp"
#
#
# 	if avgξ != 0 && f′ != 0.5
# 		varξ = avgξ-avgξ^2
# 	end
#
#     pars = Params(ϵ, ∂ϵ, ψ, maxiters, verb)
#     results = Any[]
#
#     op = OrderParams(qt, q0, q1, Qt, Q,	St,	S, Mt, M, avgWt, avgW, q̂t, q̂0, q̂1, Q̂t, Q̂, Ŝt, Ŝ, D̂)
#     ep = ExtParams(α, D, y, avgξ, varξ, f′, K, ls)
#
#     for x in xlist
#         ylist = Any[]
#         Σlist = Any[]
#         ok = false
#
# 		if with_D
# 			ep.D = x
# 		else
# 			op.Q̂ = x
# 		end
#
#         if extrapol
#             extrapolate_params!(ep, op)
#         end
#         ###PRIMO TENTATIVO
#         println("@@@ FIRST TRIAL SIGMA0")
#         println("@@@  y=$(ep.y)")
#         println(ep)
#         println(op)
#
# 		if with_D
# 			ok = converge_D!(op, ep, pars)
# 		else
# 			ok = converge!(op, ep, pars)
# 		end
#         assert(ok)
#
#         tf = all_therm_func(ep, op)
#         println(tf)
#         push!(ylist, ep.y)
#         push!(Σlist, tf.Σ)
#         absSigma = abs(tf.Σ)
#
#         println("\n @@@  y=$(ep.y)  Σ=$(tf.Σ) \n")
#
#         ###SECOND TENTATIVO
#         if absSigma > tol
#             println("@@@SECOND TRIAL SIGMA0")
#             ep.y += ep.y* tf.Σ * dy
#             println("@@@ new  y=$(ep.y)")
# #             ep = ExtParams(α, β, y, fŜ(op.q̂t, op.q̂0, op.q̂1, op.Ŝt, op.Ŝ,  y))
#             if with_D
# 				ok = converge_D!(op, ep, pars)
# 			else
# 				ok = converge!(op, ep, pars)
# 			end
#             assert(ok)
#             tf = all_therm_func(ep, op)
#             println(tf)
#             push!(ylist, ep.y)
#             push!(Σlist, tf.Σ)
#             absSigma = abs(tf.Σ)
#             println("\n @@@  y=$(ep.y)   Σ=$(tf.Σ) \n")
#         end
#
#         ###ALTRI  TENTATIVI
#         trial = 3
#         while absSigma > tol
#             println("TRIAL $(trial)  SIGMA0")
#             dummyfile = string("dummy" ,string(rand()),".dat")
#             rf = open(dummyfile, "w")
#             m = max(1, length(ylist)-4)
#             for i=m:length(ylist)
#     #             println("$Ŝ $f0")
#                 println(rf, "$(ylist[i]) $(Σlist[i])")
#             end
#             close(rf)
#             s = 0
#             if trial == 3
#                 s = -(ylist[2]*Σlist[1] - ylist[1]*Σlist[2])/(Σlist[2]-Σlist[1])
#             elseif trial == 4
#                 s = float(readall(`gnuplot -e "filename='$dummyfile'" interpolation-quad.gpl `))
#             else
#                 s = float(readall(`gnuplot -e "filename='$dummyfile'" interpolation.gpl `))
#             end
#             run(`rm $dummyfile`)
#             if smallsteps && abs(s - ep.y) >  ep.y *abs(tf.Σ) * dy
#                 ep.y += sign(s - ep.y) * ep.y* abs(tf.Σ) * dy
#             else
#                 ep.y = s
#             end
#
#             println("@@@ new  y=$(ep.y)")
# #             ep = ExtParams(α, β, y, fŜ(op.q̂t, op.q̂0, op.q̂1, op.Ŝt, op.Ŝ,  y))
#             if with_D
# 				ok = converge_D!(op, ep, pars)
# 			else
# 				ok = converge!(op, ep, pars)
# 			end
#             assert(ok)
#             tf = all_therm_func(ep, op)
#             println(tf)
#             println("\n @@@  y=$(ep.y)   Σ=$(tf.Σ) \n")
#             push!(ylist, ep.y)
#             push!(Σlist, tf.Σ)
#             absSigma = abs(tf.Σ)
#             trial += 1
#         end
#
#         push!(results, (ok, deepcopy(op), deepcopy(ep), deepcopy(tf)))
#         verb > 0 &&   println(tf)
#         if ok
#             println("\n\n@@@ SUCCESS SIGMA0\n\n")
#             while isreadable(lockfile)
#                 info("waiting for lockfile release [$lockfile]...")
#                 sleep(2)
#             end
#             try
#                 run(`touch $lockfile`)
#                 open(resfile, "a") do rf
#                     veryshortshow(rf, ep); print(rf, " ")
#                     veryshortshow(rf, tf); print(rf, " ")
#                     veryshortshow(rf, op); print(rf," ")
#                     println(rf, "X ", n1," ",n2," ", n3,)
#                 end
#             finally
#                 isreadable(lockfile) && rm(lockfile)
#             end
#         else
#             @warn("@@@ FAILURE SIGMA0")
#         end
#         ok || break
#         #Σ > 0 || break
#     end
# end



# function readparams1RSB(; file = nothing, row = 0)
#
#     if file == nothing
#         file = "results_1RSB_LargeY_FiniteT_generalized.new.txt"
#     end
#     a = readdlm(file)
#     r = []
#     if row <= 0
#         r = a[end+row,:]
#     else
#         r = a[row,:]
#     end
#
#     ep = ExtParams(r[5], r[2:4]..., r[6],[0:4;],r[7])
#     op = OrderParams(r[9],r[10],r[10]+1e-6, r[11]
#                     ,r[12]
#                     ,r[14],r[15],r[15]+1e-6, r[16]
#                     ,r[17],1)
#     return ep,op
# end

end ## module
