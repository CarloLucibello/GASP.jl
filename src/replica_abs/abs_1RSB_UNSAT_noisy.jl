module PhaseRetr

using LittleScienceTools.Roots
using QuadGK
using AutoGrad
using Cubature
using IterTools: product
import LsqFit: curve_fit
using FastGaussQuadrature
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 12.0
const dx = 0.02

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., atol=1e-7, maxevals=10^7)[1]

## Cubature.jl

∫∫D(f, xmin::Vector, xmax::Vector) = hcubature(z->begin
            r = G(z[1])*G(z[2])*f(z[1],z[2])
            isfinite(r) ? r : 0.0
        end, xmin, xmax, abstol=1e-7)[1]


∫∫D(fdim, f, xmin::Vector, xmax::Vector) = hcubature(fdim, (z,y)->begin
        r .= (G(z[1]).*G(z[2])).*f(z[1],z[2])
        isfinite(r) ? r : 0.0
    end, xmin, xmax, abstol=1e-7)[1]

## Cuba.jl.
# ∫∫∫D(f, xmin::Vector, xmax::Vector) = cuhre((z,y)->begin
#             @. z = xmin + z*(xmax-xmin)
#             y[1] = G(z[1])*G(z[2])*G(z[3])*f(z[1],z[2],z[3])
#             # isfinite(r) ? r : 0.0
#         end, 3, 1,  abstol=1e-10)[1][1]*prod(xmax.-xmin)

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


let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    global gw
    gw(n::Int) = get!(s, n) do
        (x,w) = gausshermite(n)
        return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
    end
end

function ∫DD(f; n=161)
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


# ∫Dexp(f, g=z->1, int=interval) = quadgk(z->begin
#     r = logG(z) + f(z)
#     r = exp(r) * g(z)
# end, int..., abstol=1e-10, maxevals=10^7)[1]

# Numerical Derivaaive
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
    q0::Float64
    q1::Float64
    δq::Float64
    qh0::Float64
    qh1::Float64
    δqh::Float64
    ρh::Float64
    m::Float64 # parisi breaking parameter
end


function extrapolate!(op, ops::Vector{OrderParams})
    ord = length(ops) - 2 #fit order
    model(x, p) = sum(p[i+1]./ x.^i for i=0:ord)
    for i=1:length(fieldnames(ops[1]))
        p₀ = [getfield(op,i); zeros(ord)]
        y = curve_fit(model, 1:length(ops), [getfield(o,i) for o in ops], p₀).param[1]
        setfield!(op, i, y)
    end
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
function Gi(op, ep)
    @extract op: q0 q1 δq qh0 qh1 δqh ρh m
    @extract ep: ρ
    (q1*δqh - 2*ρ*ρh - δq*qh1 + q0*qh0*m - q1*qh1*m)/2
end

function ∂m_Gi(op, ep)
    @extract op: q0 q1 δq qh0 qh1 δqh ρh
    @extract ep: ρ
    (q0*qh0 - q1*qh1) / 2
end

#### ENTROPIC TERM ####

function Gs(op)
    @extract op: qh0 qh1 δqh ρh m
    0.5*((Power(ρh,2) + qh0)/(δqh + (qh0 - qh1)*m) + Log(δqh)/m - Log(δqh + (qh0 - qh1)*m)/m)
end

function ∂m_Gs(op)
    @extract op: qh0 qh1 δqh ρh m
    (-((qh0 - qh1)/(m*(δqh + m*(qh0 - qh1)))) -
    ((qh0 - qh1)*(qh0 + Power(ρh,2)))/Power(δqh + m*(qh0 - qh1),2) -
    Log(δqh)/Power(m,2) + Log(δqh + m*(qh0 - qh1))/Power(m,2))/2
end

#### ENERGETIC TERM ####


function argGe(y, h, δq)
    ### find min of 1/2 u^2 + (y - abs(√δq*u + h))^2
    # u* = 2√δq*(y-|z0|)*sign(h) / (1+2δq) : -h/√δq
    abs(h) > - 2δq*y ? (y - abs(h))^2 / (1 + 2δq) : h^2/(2δq) + y^2
end
function ∂z_argGe(y, h, δq)
    -2sign(h)*(y - abs(h)) / (1 + 2δq)
end
function ∂δq_argGe(y, h, δq)
    -2*(y - abs(h))^2 / (1 + 2δq)^2
end
function Ge₀(y, h, q10, δq, m)
    a = sqrt((1 + 2*δq)*(1 + 2*m*q10 + 2*δq))
    Z1 = H(-((2*m*q10*y - h*(1 + 2*δq))/
        (sqrt(q10)*a)))*exp(-(m*Power(h + y,2))/(1 + 2*m*q10 + 2*δq))
    Z2 = H(-((2*m*q10*y + h*(1 + 2*δq))/
        (sqrt(q10)*a)))*exp(-(m*Power(h - y,2))/(1 + 2*m*q10 + 2*δq))
    Z = (1 + 2*δq)*(Z1 + Z2) / a
    1/m * log(Z)
end

function Ge(op, ep)
    @extract op: q0 q1 δq m
    @extract ep: ρ Δ

    # ∫D(u0->∫D(z0->begin
    #     Ge₀(abs(u0), ρ*u0 + √(q0 - ρ^2)*z0, q1-q0, δq, m)
    # end))
    if Δ == 0
        ge = ∫∫D((u0,z0)->begin
            Ge₀(abs(u0), ρ*u0 + √(q0 - ρ^2)*z0, q1-q0, δq, m)
        end)
    else
        a1 = √(Δ+(1-ρ^2/q0))
        b1 = (ρ/√q0)
        c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
        d1 = (√(1-ρ^2/q0)/√Δ)
        # ge = 2∫D(z0->∫D(y->begin
        ge = 2∫∫D((z0,y)->begin
            argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * H(-(c1*z0 + d1*y))
        end)
    end
    return ge
end

function argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, m)
    a = sqrtq0_z0
    c = sqrt_q1_q0
    c2m = c^2*m
    d = 1+2δq
    b = (d+2c2m)
    g = (c*√(1+c2m/δq))
    g2 = (c*√(1+2c2m/d))

    1/m*log(
        if y < 0
            ### from cusp u*= -(sqrt_q1_q0*z1 + sqrtq0_z0)/√δq
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (H((a + 2*(δq + c2m)*y)/g) -
            H((a - 2*(δq + c2m)*y)/g)) / √(1+c2m/δq) +
            ### from minimum u*= 2√δq*(y-|sqrt_q1_q0*z1 + sqrtq0_z0|)*sign(sqrt_q1_q0*z1 + sqrtq0_z0) / (1+2δq)
          ( exp(-m*(a-y)^2/b) * H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d)

        else
            ### from minimum u*= 2√δq*(y-|sqrt_q1_q0*z1 + sqrtq0_z0|)*sign(sqrt_q1_q0*z1 + sqrtq0_z0) / (1+2δq)
          ( exp(-m*(a-y)^2/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
            exp(-m*(a+y)^2/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d)
        end
        )
end

function ∂1_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, m)
    a = sqrtq0_z0
    c = sqrt_q1_q0
    c2m = c^2*m
    d = 1+2δq
    b = (d+2c2m)
    g = (c*√(1+c2m/δq))
    g2 = (c*√(1+2c2m/d))

    1/m*(
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (
            (-2m*y) * (H((a + 2*(δq + c2m)*y)/g) - H((a-2*(δq + c2m)*y)/g)) +
            (-G((a + 2*(δq + c2m)*y)/g) - G((a - 2*(δq + c2m)*y)/g)) * 2*(δq + c2m)/g
            ) / √(1+c2m/δq) +
          ( exp(-m*(a-y)^2/b) * (2m*(a-y)/b) *  H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * (-2m*(a+y)/b) * H( (a - 2(δq+c2m)*y) / g2) +
            (exp(-m*(a-y)^2/b) * G(-(a + 2(δq+c2m)*y) / g2) +
             exp(-m*(a+y)^2/b) * G( (a - 2(δq+c2m)*y) / g2)) * 2(δq+c2m) / g2 ) / √(1+2c2m/d)
        else
          ( exp(-m*(a-y)^2/b) * (2m*(a-y)/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
            exp(-m*(a+y)^2/b) * (-2m*(a+y)/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d) +
            ( exp(-m*(a-y)^2/b) * G(-(a*d + 2c2m*y) / (c*√(d*b))) +
              exp(-m*(a+y)^2/b) * G( (a*d - 2c2m*y) / (c*√(d*b))) ) * ( 2c2m / (c*√(d*b))) / √(1+2c2m/d)
        end
        ) /
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (H((a + 2*(δq + c2m)*y)/g) -
            H((a - 2*(δq + c2m)*y)/g)) / √(1+c2m/δq) +
          ( exp(-m*(a-y)^2/b) * H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d)

        else
          ( exp(-m*(a-y)^2/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
            exp(-m*(a+y)^2/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d)
        end
end

function ∂2_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, m)
    a = sqrtq0_z0
    c = sqrt_q1_q0
    c2m = c^2*m
    d = 1+2δq
    b = (d+2c2m)
    g = (c*√(1+c2m/δq))
    g2 = (c*√(1+2c2m/d))

    1/m*(
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (
            (-2m*a/(2δq*(1+c2m/δq))) * (H((a + 2*(δq + c2m)*y)/g) - H((a-2*(δq + c2m)*y)/g)) +
            (-G((a + 2*(δq + c2m)*y)/g) + G((a - 2*(δq + c2m)*y)/g)) / g
            ) / √(1+c2m/δq) +
          ( exp(-m*(a-y)^2/b) * (-2m*(a-y)/b) *  H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * (-2m*(a+y)/b) * H( (a - 2(δq+c2m)*y) / g2) +
            (exp(-m*(a-y)^2/b) * G(-(a + 2(δq+c2m)*y) / g2) -
             exp(-m*(a+y)^2/b) * G( (a - 2(δq+c2m)*y) / g2)) / g2 ) / √(1+2c2m/d)
        else
          ( exp(-m*(a-y)^2/b) * (-2m*(a-y)/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
            exp(-m*(a+y)^2/b) * (-2m*(a+y)/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d) +
            ( exp(-m*(a-y)^2/b) * G(-(a*d + 2c2m*y) / (c*√(d*b))) -
              exp(-m*(a+y)^2/b) * G( (a*d - 2c2m*y) / (c*√(d*b))) ) * (1 / (c*√(d*b))) / √(1+2c2m/d)
        end
        ) /
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (H((a + 2*(δq + c2m)*y)/g) -
            H((a - 2*(δq + c2m)*y)/g)) / √(1+c2m/δq) +
          ( exp(-m*(a-y)^2/b) * H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d)

        else
          ( exp(-m*(a-y)^2/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
            exp(-m*(a+y)^2/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d)
        end
end

function ∂3_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, m)
    a = sqrtq0_z0
    c = sqrt_q1_q0
    c2m = c^2*m
    d = 1+2δq
    b = (d+2c2m)
    g = (c*√(1+c2m/δq))
    g2 = (c*√(1+2c2m/d))

    1/m*(
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) *
            ((4*m^2*a^2/(2δq*(1+c2m/δq))^2*c) * (H((a + 2*(δq + c2m)*y)/g) - H((a-2*(δq + c2m)*y)/g)) +
            - G((a + 2*(δq + c2m)*y)/g) * (4*y*c*m/g - (a + 2*(δq + c2m)*y)/g^2 * (√(1+c2m/δq) + c/√(1+c2m/δq)*m*c/δq)) +
            + G((a - 2*(δq + c2m)*y)/g) * (-4*y*c*m/g - (a - 2*(δq + c2m)*y)/g^2 * (√(1+c2m/δq) + c/√(1+c2m/δq)*m*c/δq))
            ) / √(1+c2m/δq) +
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (H((a + 2*(δq + c2m)*y)/g) -
            H((a - 2*(δq + c2m)*y)/g)) / √(1+c2m/δq) * (-1/(1+c2m/δq)*m*c/δq) +
            ( exp(-m*(a-y)^2/b) * (4*c*m^2*(a-y)^2/b^2) * H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * (4*c*m^2*(a+y)^2/b^2) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d) -
            (exp(-m*(a-y)^2/b) * G(-(a + 2(δq+c2m)*y) / g2) * (-4c*m*y/g2 + (a+2(δq+c2m)*y)/g2^2 * (√(1+2c2m/d) + c/√(1+2c2m/d)*2*c*m/d))  +
             exp(-m*(a+y)^2/b) * G( (a - 2(δq+c2m)*y) / g2) * (-4c*m*y/g2 - (a-2(δq+c2m)*y)/g2^2 * (√(1+2c2m/d) + c/√(1+2c2m/d)*2*c*m/d))
             ) / √(1+2c2m/d) +
             ( exp(-m*(a-y)^2/b) * H(-(a + 2(δq+c2m)*y) / g2) +
               exp(-m*(a+y)^2/b) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d) * (-1/(1+2c2m/d)*2*c*m/d)
        else
            ( exp(-m*(a-y)^2/b) * (4c*m^2*(a-y)^2/b^2) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
              exp(-m*(a+y)^2/b) * (4c*m^2*(a+y)^2/b^2) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d) +
            ( -exp(-m*(a-y)^2/b) * G(-(a*d + 2c2m*y) / (c*√(d*b))) * (-(4c*m*y) / (c*√(d*b)) + (a*d + 2c2m*y) / (c*√(d*b))^2 * (√(d*b) + 2*c/√(d*b)*d*c*m)) -
              exp(-m*(a+y)^2/b) * G( (a*d - 2c2m*y) / (c*√(d*b))) * (-(4c*m*y) / (c*√(d*b)) - (a*d - 2c2m*y) / (c*√(d*b))^2 * (√(d*b) + 2*c/√(d*b)*d*c*m))) / √(1+2c2m/d) +
              ( exp(-m*(a-y)^2/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
                exp(-m*(a+y)^2/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d) * (-1/(1+2c2m/d)*2*c*m/d)
        end
        ) /
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (H((a + 2*(δq + c2m)*y)/g) -
            H((a - 2*(δq + c2m)*y)/g)) / √(1+c2m/δq) +
          ( exp(-m*(a-y)^2/b) * H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d)

        else
          ( exp(-m*(a-y)^2/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
            exp(-m*(a+y)^2/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d)
        end
end

function ∂4_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, m)
    a = sqrtq0_z0
    c = sqrt_q1_q0
    c2m = c^2*m
    d = 1+2δq
    b = (d+2c2m)
    g = (c*√(1+c2m/δq))
    g2 = (c*√(1+2c2m/d))

    1/m*(
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (
            (2*m*a^2/(2δq*(1+c2m/δq))^2) * (H((a + 2*(δq + c2m)*y)/g) - H((a-2*(δq + c2m)*y)/g)) +
            - G((a + 2*(δq + c2m)*y)/g) * (2*y/g - (a + 2*(δq + c2m)*y)/g^2 * (1/2*c/√(1+c2m/δq)*(-c2m/δq^2))) +
            + G((a - 2*(δq + c2m)*y)/g) * (-2*y/g - (a - 2*(δq + c2m)*y)/g^2 * (1/2*c/√(1+c2m/δq)*(-c2m/δq^2)))
            ) / √(1+c2m/δq) +
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (H((a + 2*(δq + c2m)*y)/g) -
            H((a - 2*(δq + c2m)*y)/g)) / √(1+c2m/δq) * (-1/2/(1+c2m/δq)*(-c2m/δq^2)) +
            ( exp(-m*(a-y)^2/b) * (2*m*(a-y)^2/b^2) * H(-(a + 2(δq+c2m)*y) / g2) +
              exp(-m*(a+y)^2/b) * (2*m*(a+y)^2/b^2) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d) -
            (exp(-m*(a-y)^2/b) * G(-(a + 2(δq+c2m)*y) / g2) * (-2*y/g2 + (a+2(δq+c2m)*y)/g2^2 * (-c/√(1+2c2m/d)*2*c2m/d^2))  +
             exp(-m*(a+y)^2/b) * G( (a - 2(δq+c2m)*y) / g2) * (-2*y/g2 - (a-2(δq+c2m)*y)/g2^2 * (-c/√(1+2c2m/d)*2*c2m/d^2))
             ) / √(1+2c2m/d) +
             ( exp(-m*(a-y)^2/b) * H(-(a + 2(δq+c2m)*y) / g2) +
               exp(-m*(a+y)^2/b) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d) * (1/(1+2c2m/d)*2*c2m/d^2)
        else
            ( exp(-m*(a-y)^2/b) * (2m*(a-y)^2/b^2) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
              exp(-m*(a+y)^2/b) * (2m*(a+y)^2/b^2) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d) +
            ( -exp(-m*(a-y)^2/b) * G(-(a*d + 2c2m*y) / (c*√(d*b))) * (-(2a) / (c*√(d*b)) + (a*d + 2c2m*y) / (c*√(d*b))^2 * (1/2*c/√(d*b)*(2d+2b))) -
              exp(-m*(a+y)^2/b) * G( (a*d - 2c2m*y) / (c*√(d*b))) * ((2a) / (c*√(d*b)) - (a*d - 2c2m*y) / (c*√(d*b))^2 * (1/2*c/√(d*b)*(2d+2b)))) / √(1+2c2m/d) +
              ( exp(-m*(a-y)^2/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
                exp(-m*(a+y)^2/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d) * (1/(1+2c2m/d)*2c2m/d^2)
        end
        ) /
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (H((a + 2*(δq + c2m)*y)/g) -
            H((a - 2*(δq + c2m)*y)/g)) / √(1+c2m/δq) +
          ( exp(-m*(a-y)^2/b) * H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d)
        else
          ( exp(-m*(a-y)^2/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
            exp(-m*(a+y)^2/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d)
        end
end

function ∂5_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, m)
    a = sqrtq0_z0
    c = sqrt_q1_q0
    c2m = c^2*m
    d = 1+2δq
    b = (d+2c2m)
    g = (c*√(1+c2m/δq))
    g2 = (c*√(1+2c2m/d))

    1/m*(
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (
            (-a^2/(2δq*(1+c2m/δq)) + m*a^2/(2δq*(1+c2m/δq))^2*2c^2-y^2) * (H((a + 2*(δq + c2m)*y)/g) - H((a-2*(δq + c2m)*y)/g)) +
            - G((a + 2*(δq + c2m)*y)/g) * (2*y*c^2/g - (a + 2*(δq + c2m)*y)/g^2 * (1/2*c^3/√(1+c2m/δq)/δq)) +
            + G((a - 2*(δq + c2m)*y)/g) * (-2*y*c^2/g - (a - 2*(δq + c2m)*y)/g^2 * (1/2*c^3/√(1+c2m/δq)/δq))
            ) / √(1+c2m/δq) +
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (H((a + 2*(δq + c2m)*y)/g) -
            H((a - 2*(δq + c2m)*y)/g)) / √(1+c2m/δq) * (-1/2/(1+c2m/δq)*c^2/δq) +

            ( exp(-m*(a-y)^2/b) * (-(a-y)^2/b + m*2*c^2*(a-y)^2/b^2) * H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * (-(a+y)^2/b + m*2*c^2*(a+y)^2/b^2) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d) -
            (exp(-m*(a-y)^2/b) * G(-(a + 2(δq+c2m)*y) / g2) * (-2c^2*y/g2 + (a+2(δq+c2m)*y)/g2^2 * (c^3/√(1+2c2m/d)/d))  +
             exp(-m*(a+y)^2/b) * G( (a - 2(δq+c2m)*y) / g2) * (-2c^2*y/g2 - (a-2(δq+c2m)*y)/g2^2 * (c^3/√(1+2c2m/d)/d))
             ) / √(1+2c2m/d) +
             ( exp(-m*(a-y)^2/b) * H(-(a + 2(δq+c2m)*y) / g2) +
               exp(-m*(a+y)^2/b) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d) * (-1/(1+2c2m/d)*c^2/d)
        else
            ( exp(-m*(a-y)^2/b) * (-(a-y)^2/b + m*2*c^2*(a-y)^2/b^2) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
              exp(-m*(a+y)^2/b) * (-(a+y)^2/b + m*2*c^2*(a+y)^2/b^2) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d) +
            ( -exp(-m*(a-y)^2/b) * G(-(a*d + 2c2m*y) / (c*√(d*b))) * (-(2c^2*y) / (c*√(d*b)) + (a*d + 2c2m*y) / (c*√(d*b))^2 * (c/√(d*b)*d*c^2)) -
              exp(-m*(a+y)^2/b) * G( (a*d - 2c2m*y) / (c*√(d*b))) * (-(2c^2*y) / (c*√(d*b)) - (a*d - 2c2m*y) / (c*√(d*b))^2 * (c/√(d*b)*d*c^2))) / √(1+2c2m/d) +
              ( exp(-m*(a-y)^2/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
                exp(-m*(a+y)^2/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d) * (-1/(1+2c2m/d)*c^2/d)
        end
        ) /
        if y < 0
            exp(-m*a^2/(2δq*(1+c2m/δq))-m*y^2) * (H((a + 2*(δq + c2m)*y)/g) -
            H((a - 2*(δq + c2m)*y)/g)) / √(1+c2m/δq) +
          ( exp(-m*(a-y)^2/b) * H(-(a + 2(δq+c2m)*y) / g2) +
            exp(-m*(a+y)^2/b) * H( (a - 2(δq+c2m)*y) / g2) ) / √(1+2c2m/d)

        else
          ( exp(-m*(a-y)^2/b) * H(-(a*d + 2c2m*y) / (c*√(d*b))) +
            exp(-m*(a+y)^2/b) * H( (a*d - 2c2m*y) / (c*√(d*b))) ) / √(1+2c2m/d)
        end +
    -1/m * argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, m)
end

function ∂q0_Ge(op, ep)
    @extract op: q0 q1 δq m
    @extract ep: ρ Δ

    if Δ == 0
        op.q0 += 1e-6
        ge1 = Ge(op,ep)
        op.q0 -= 2e-6
        ge2 = Ge(op,ep)
        op.q0 += 1e-6
        return (ge1 - ge2) / 2e-6
    else
        a1 = √(Δ+(1-ρ^2/q0))
        da1 = 1/(2*a1)*ρ^2/q0^2
        b1 = (ρ/√q0)
        db1 = -1/2*b1/q0
        c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
        dc1 = -1/2*c1/q0 -1/2*ρ/√q0/√(1/(1-ρ^2/q0)+ 1/Δ)*1/(1-ρ^2/q0)^2*ρ^2/q0^2
        d1 = (√(1-ρ^2/q0)/√Δ)
        dd1 = 1/2/√(1-ρ^2/q0)/√Δ*ρ^2/q0^2
        # ge = 2∫DD(z0->∫DD(y->begin
        ge = 2∫∫D((z0,y)->begin
            ∂1_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * (da1*y + db1*z0) * H(-(c1*z0 + d1*y)) +
            ∂2_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * (1/2*z0/√q0) * H(-(c1*z0 + d1*y)) +
            ∂3_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * (-1/(2*√(q1-q0))) * H(-(c1*z0 + d1*y)) +
            argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * G(-(c1*z0 + d1*y)) * (dc1*z0 + dd1*y)
        end)
    end
end

function ∂q1_Ge(op, ep)
    @extract op: q0 q1 δq m
    @extract ep: ρ Δ

    if Δ == 0
        op.q1 += 1e-6
        ge1 = Ge(op,ep)
        op.q1 -= 2e-6
        ge2 = Ge(op,ep)
        op.q1 += 1e-6
        return (ge1 - ge2) / 2e-6
    else
        a1 = √(Δ+(1-ρ^2/q0))
        b1 = (ρ/√q0)
        c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
        d1 = (√(1-ρ^2/q0)/√Δ)
        # ge = 2∫DD(z0->∫DD(y->begin
        ge = 2∫∫D((z0,y)->begin
            ∂3_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * (1/(2*√(q1-q0))) * H(-(c1*z0 + d1*y))
        end)
    end
end

function ∂δq_Ge(op, ep)
    @extract op: q0 q1 δq m
    @extract ep: ρ Δ

    if Δ == 0
        op.δq += 1e-6
        ge1 = Ge(op,ep)
        op.δq -= 2e-6
        ge2 = Ge(op,ep)
        op.δq += 1e-6
        return (ge1 - ge2) / 2e-6
    else
        a1 = √(Δ+(1-ρ^2/q0))
        b1 = (ρ/√q0)
        c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
        d1 = (√(1-ρ^2/q0)/√Δ)
        # ge = 2∫DD(z0->∫DD(y->begin
        ge = 2∫∫D((z0,y)->begin
            ∂4_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * H(-(c1*z0 + d1*y))
        end)
    end
end

function ∂ρ_Ge(op, ep)
    @extract op: q0 q1 δq m
    @extract ep: ρ Δ
    if Δ == 0
        ep.ρ += 1e-6
        ge1 = Ge(op,ep)
        ep.ρ -= 2e-6
        ge2 = Ge(op,ep)
        ep.ρ += 1e-6
        return (ge1 - ge2) / 2e-6
    else
        a1 = √(Δ+(1-ρ^2/q0))
        da1 = -1/(a1)*ρ/q0
        b1 = (ρ/√q0)
        db1 = 1/√q0
        c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
        dc1 = c1/ρ + ρ/√q0/√(1/(1-ρ^2/q0)+ 1/Δ)*1/(1-ρ^2/q0)^2*ρ/q0
        d1 = (√(1-ρ^2/q0)/√Δ)
        dd1 = -1/√(1-ρ^2/q0)/√Δ*ρ/q0
        # ge = 2∫DD(z0->∫DD(y->begin
        ge = 2∫∫D((z0,y)->begin
            ∂1_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * (da1*y + db1*z0) * H(-(c1*z0 + d1*y)) +
            argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * G(-(c1*z0 + d1*y)) * (dc1*z0 + dd1*y)
        end)
    end
end

function ∂m_Ge(op, ep)
    @extract op: q0 q1 δq m
    @extract ep: ρ Δ
    if Δ == 0
        op.m += 1e-6
        ge1 = Ge(op,ep)
        op.m -= 2e-6
        ge2 = Ge(op,ep)
        op.m += 1e-6
        return (ge1 - ge2) / 2e-6
    else
        a1 = √(Δ+(1-ρ^2/q0))
        b1 = (ρ/√q0)
        c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
        d1 = (√(1-ρ^2/q0)/√Δ)
        # ge = 2∫DD(z0->∫DD(y->begin
        ge = 2∫∫D((z0,y)->begin
            ∂5_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, m) * H(-(c1*z0 + d1*y))
        end)
    end
end

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    Gi(op, ep) + Gs(op) + ep.α*Ge(op, ep)
end

## Thermodinamic functions
# The energy of the pure states selected by x
# E = -∂(m*ϕ)/∂m
# if working at fixed x. If x is optimized Σ=0 and
# E = -ϕ
# so this formula is valid both at fixed and at optimized x
function all_therm_func(op::OrderParams, ep::ExtParams)
    ϕ = free_entropy(op, ep)
    E = -ϕ - op.m * im_fun(op, ep, op.m)
    Σ = op.m*(ϕ + E)
    return ThermFunc(ϕ, Σ, E)
end

#################  SADDLE POINT  ##################
fqh0(op, ep) = -2/op.m * ep.α * ∂q0_Ge(op, ep)
fqh1(op, ep) = 2ep.α * ∂δq_Ge(op, ep)
fδqh(op, ep) = op.qh1*op.m -2ep.α * ∂q1_Ge(op, ep)
fρh(op, ep) = ep.α * ∂ρ_Ge(op, ep)

function fq0(op)
    @extract op: qh0 qh1 δqh ρh m
    (qh0 + ρh^2) / (δqh + m*(qh0 - qh1))^2
end

function fδq(op)
    @extract op: qh0 qh1 δqh ρh m q1
    - m*q1 + 1/(δqh + m*(qh0 - qh1)) + (m*(qh0 + Power(ρh,2)))/Power(δqh + m*(qh0 - qh1),2)
end
function fq1(op)
    @extract op: qh0 qh1 δqh ρh m q1
    -(1/(δqh*m) - (qh0 + Power(ρh,2))/Power(δqh + (qh0 - qh1)*m,2) -
      1/(m*(δqh + (qh0 - qh1)*m)))
end
function fρ(op)
    @extract op: qh0 qh1 δqh ρh m q1
    ρh/(δqh + m*(qh0 - qh1))
end

function iρh(op, ep)
    @extract op: qh0 qh1 δqh m q1
    (true, ep.ρ*(δqh + m*(qh0 - qh1)))
end
function iδqh_fun(δq, op)
    @extract op: qh0 qh1 δqh ρh m q1
    0.5*q1 + (1/(δqh*m) - 1/(m*(δqh + m*(qh0 - qh1))) -
    (qh0 + Power(ρh,2))/Power(δqh + m*(qh0 - qh1),2))/2.
end

function iδqh(op, δqh₀, atol=1e-10)
    ok, δqh, it, normf0 = findroot(δqh -> iδqh_fun(δqh, op), δqh₀, NewtonMethod(atol=atol))
    ok || error("iδqh failed: iδqh=$(δqh), it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, δqh
end


function im_fun(op::OrderParams, ep::ExtParams, m)
    @extract op: q0 q1 δq qh0 qh1 δqh ρh
    @extract ep: α ρ
    op.m = m # for Ge
    ∂m_Gi(op, ep) + ∂m_Gs(op) + α*∂m_Ge(op, ep)
end

function im(op::OrderParams, ep::ExtParams, m₀, atol=1e-4)
    ok, m, it, normf0 = findroot(m -> im_fun(op, ep, m), m₀, NewtonMethod(atol=atol))
    ok || error("im failed: m=$m, it=$it, normf0=$normf0")
    return ok, m
end


###############################
function fhats_slow(op, ep)
    qh0 = qh1 = δqh = 0
    @sync begin
        qh0 = @spawn fqh0(op, ep)
        qh1 = @spawn fqh1(op, ep)
        δqh = @spawn fδqh(op, ep)
        ρh = @spawn fρh(op, ep)
    end
    return fetch(qh0), fetch(qh1), (fetch(qh1)-op.qh1)*op.m + fetch(δqh), fetch(ρh)
end

function fix_inequalities!(op, ep)
    if op.q0 < ep.ρ^2
        op.q0 = ep.ρ^2 + rand()*1e-4
    end
    if op.δq < 0
        op.δq = rand()
    end
    if op.q0 < 0
        op.q0 = rand()*1e-4
    end
    if op.q1 < op.q0
        op.q1 = op.q0 + rand()*1e-4
    end
    if op.δqh + (op.qh0 - op.qh1)*op.m < 0
        op.δqh = (op.qh0 - op.qh1)*op.m + rand()*1e-4
    end
end

function converge!(op::OrderParams, ep::ExtParams, pars::Params;
                   testρ = false,
        fixρ=true, fixnorm=true, fixm=true, extrap=-1)
    @extract pars: maxiters verb ϵ ψ

    Δ = Inf
    ok = false
    fix_inequalities!(op, ep)

    ops = Vector{OrderParams}() # keep some history and extrapolate for quicker convergence
    for it = 1:maxiters
        Δ = 0.0
        ok = oki = true
        verb > 1 && println("it=$it")


        qh0, qh1, δqh, ρh = fhats_slow(op, ep)

        @update  op.qh0    identity       Δ ψ verb  qh0
        @update  op.qh1    identity       Δ ψ verb  qh1
        if fixnorm
            @updateI op.δqh ok   iδqh     Δ ψ verb  op ep
        else
            @update  op.δqh    identity     Δ ψ verb  δqh
        end
        if fixρ
            @updateI op.ρh oki   iρh   Δ ψ verb  op ep
            ok &= oki
        else
            @update  op.ρh  identity    Δ ψ verb  ρh
        end

        # fix_inequalities_hat!(op, ep)
        fix_inequalities!(op, ep)

        @update op.q0   fq0       Δ ψ verb  op
        if !fixnorm
            @update op.q1   fq1     Δ ψ verb  op
        end
        @update op.δq   fδq       Δ ψ verb  op
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op
        end

        if !fixm
            @updateI op.m oki   im    Δ ψ verb  op ep op.m
            ok &= oki
        end

        fix_inequalities!(op, ep)

        verb > 1 && println(" Δ=$Δ\n")
        verb > 4 && it%5==0 && (println(ep);println(all_therm_func(op, ep));println(op))

        @assert isfinite(Δ)
        ok &= Δ < ϵ
        if ok || (testρ && ep.ρ < 1e-5)
            println(op)
            break
        end

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
    qh0 = 0.048579636269663,
  qh1 = 0.05011573365196532,
  δqh = 1.858836854048346,
  ρh = 1.5085044105775007,
  q0 = 0.9024632620210173,
  q1 = 0.902978204529109,
  δq = 0.5379708271987909,
        ρ=0, m=1.,
        α=2.0,
        ϵ=1e-4, maxiters=100000, verb=3, ψ=0.,
        fixρ=true, fixnorm=true, fixm=true, extrap=-1
    )
    op = OrderParams(q0,q1,δq,qh0,qh1,δqh,ρh,m)
    ep = ExtParams(α, ρ, Δ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars,
        fixρ=fixρ,fixnorm=fixnorm,extrap=extrap,fixm=fixm,extrap=extrap)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
    q0=0.9985202029186012,q1=0.9985204772596665,δq=0.25572539819580564,qh0=0.00793938024957339,qh1=0.00794357534356902,δqh=3.91044459038955,ρh=3.906521573102456,m=3.0,
        α=3.0,ρ=0.93,Δ=0.001,
        ϵ=1e-4, maxiters=10000,verb=3, ψ=0.,targetΣ=0., maximum=false,
        kws...)

    op = OrderParams(q0,q1,δq,qh0,qh1,δqh,ρh, first(m))
    ep = ExtParams(first(α), first(ρ), first(Δ))
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; ρ=ρ,α=α,Δ=Δ,m=m, targetΣ=targetΣ, maximum=maximum, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
        α=1, ρ=1, m=1, Δ=0.1,
        resfile = "results1RSB.txt",
        fixρ=true, fixnorm=true, fixm=true, extrap=-1, targetΣ=0., maximum=false)

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    results = []
    for m in m, α in α, ρ in ρ, Δ in Δ
        fixm && (op.m = m)
        fixρ && (ep.ρ = ρ)
        ep.α = α
        ep.Δ = Δ
        println("# NEW ITER: α=$(ep.α)  ρ=$(ep.ρ)  Δ=$(ep.Δ)  m=$(op.m)")

        if fixm
            ok = converge!(op, ep, pars, fixm=true, fixρ=fixρ,fixnorm=fixnorm)
        else
            ok = maximum ? findmaximumSigma!(op, ep, pars; tol=1e-9,fixρ=fixρ,fixnorm=fixnorm) : findSigma0!(op, ep, pars; tol=pars.ϵ,fixρ=fixρ,fixnorm=fixnorm, targetΣ=targetΣ)
            # ok, m, it, normf0 = findroot(m -> begin
            #                         println("# FINDROOT m=$m")
            #                         op.m = m
            #                         ok = converge!(app, op, ep, fixm=true, fixρ=fixρ)
            #                         println(op)
            #                         println(all_therm_func(app, op, ep))
            #                         im_fun(app, op, ep, m)
            #                     end, op.m, NewtonMethod(atol=app.ϵ/10, dx=100*app.ϵ))
            # ok || error("im failed: m=$m, it=$it, normf0=$normf0")
            # op.m = m
        end
        tf = all_therm_func(op, ep)
        tf.Σ < -1e-5 && @warn("Sigma negative")
        push!(results, (ok, deepcopy(ep), deepcopy(tf), deepcopy(op)))
        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")

        # if extrap > 0 && it > extrap && it % extrap == 0
        #     extrapolate!(op, ops)
        #     empty!(ops)
        #     verb > 1 && println("# estrapolation -> $op \n")
        # end
    end
    return results
end


function span2!(op::OrderParams, ep::ExtParams, pars::Params;
        αs=-1, Δs=-1,
        resfile = "results1RSB.txt",
        line = 1,
        fixnorm=false, extrap=-1, targetΣ=0., maximum_m=2000)

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    results = []

    ep, op, tf = readparams(resfile; line=line)
    for α in αs, Δ in Δs
        α > -1 && (ep.α = α)
        Δ > -1 && (ep.Δ = Δ)
        println("# NEW ITER: α=$(ep.α)  ρ=$(ep.ρ)  Δ=$(ep.Δ)  m=$(op.m)")

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




function find_best_start_params(resfile, α, Δ, left)
    res = readdlm(resfile)
    resmap = map(i->res[i,:],1:size(res,1))
    left ? filter!(i->i[end]=="left",resmap) : filter!(i->i[end]=="right",resmap)
    c = map(i->abs(resmap[i][1]-α)/1.5 + abs(resmap[i][3]-Δ)/0.001,1:length(resmap))
    sp1 = sortperm(c)[1]
    ep = ExtParams(resmap[sp1][1:3]...)
    op = OrderParams(resmap[sp1][7:end-1]...)
    tf = ThermFunc(resmap[sp1][4:6]...)
    return ep, op, tf
end

function spanΔ(;
    q0=0.9985202029186012,q1=0.9985204772596665,δq=0.25572539819580564,qh0=0.00793938024957339,qh1=0.00794357534356902,δqh=3.91044459038955,ρh=3.906521573102456,m=3.0,
        α=3.0,ρ=0.,Δ=0.001,
        ϵ=1e-4, maxiters=10000,verb=3, ψ=0.,targetΣ=0., maximum=false,
        kws...)

    op = OrderParams(q0,q1,δq,qh0,qh1,δqh,ρh, first(m))
    ep = ExtParams(first(α), ρ, first(Δ))
    pars = Params(ϵ, ψ, maxiters, verb)
    return spanΔ!(op, ep, pars; Δs=Δ, αs=α, kws...)
end

function spanΔ!(op::OrderParams, ep::ExtParams, pars::Params;
    αs=1, Δs=1,
    resfile = "results.txt",
    resfile2 = "results2.txt",
    fixρ=true, fixm=true, targetΣ = 0., maximum=false, fixnorm=false)

    if !isfile(resfile2)
        f = open(resfile2, "w")
        println(f, "### 1:α 2:Δ 3:ρ_left 4:ρ_right 5:E_left 6:E_right")
        close(f)
    end

    ϵ = pars.ϵ
    maxiters = pars.maxiters

    results = []
    for α in αs
        ep.α = α;
        for Δ in Δs
            ρ_left = -1
            E_left = -1
            ρ_right = -1
            E_right = -1

            ep, op, _ = find_best_start_params(resfile, α, Δ, true)
            ep.α = α;
            ep.Δ = Δ;

            println(" >>>>>", ep)
            done = false
            while !done
                ep.ρ = 0.0;
                try
                    pars.ϵ = 1e-30;
                    pars.maxiters = 10
                    ok = converge!(op, ep, pars; fixρ=true,fixnorm=fixnorm,fixm=true)
                    pars.ϵ = ϵ;
                    pars.maxiters = maxiters
                    ok = findSigma0!(op, ep, pars; tol=pars.ϵ,fixρ=true,fixnorm=fixnorm)
                catch
                    break
                end
                tf = all_therm_func(op, ep)
                open(resfile, "a") do rf
                    println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op), " left")
                end
                E1 = tf.E

                ep.ρ = 1e-5
                pars.ϵ = 1e-30;
                pars.maxiters = 30
                ok = converge!(op, ep, pars; fixρ=false, testρ = true,fixnorm=fixnorm,fixm=true)
                pars.ϵ = ϵ;
                pars.maxiters = maxiters
                if ep.ρ < 1e-5
                    ρ_left = 0
                    E_left = E1
                else
                    if ok
                        open(resfile, "a") do rf
                            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op), " right")
                        end
                    end
                    ρ_left = 10
                    E_left = -1
                end
                done = true
            end

            # ep, op, _ = find_best_start_params(resfile, α, Δ, false)
            # ep.α = α;
            # ep.Δ = Δ;
            #
            # done= false
            # while !done
            #     try
            #         pars.ϵ = 1e-30;
            #         pars.maxiters = 10
            #         ok = converge!(op, ep, pars; fixρ=true,fixnorm=fixnorm,fixm=true)
            #         pars.ϵ = ϵ;
            #         pars.maxiters = maxiters
            #         ok = findSigma0!(op, ep, pars; tol=pars.ϵ,fixρ=true,fixnorm=fixnorm)
            #     catch
            #         break
            #     end
            #     tf = all_therm_func(op, ep)
            #     open(resfile, "a") do rf
            #         println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op), " right")
            #     end
            #
            #     ok = findSigma0!(op, ep, pars; tol=pars.ϵ,fixρ=false,fixnorm=fixnorm)
            #     tf = all_therm_func(op, ep)
            #     !ok && break
            #     ρ_right = ep.ρ
            #     E_right = tf.E
            #     done = true
            # end

            push!(results, (ep.α, ep.Δ, ρ_left, ρ_right, E_left, E_right))
            open(resfile2, "a") do rf
                println(rf, ep.α, " ", ep.Δ, " ", ρ_left, " ", ρ_right, " ", E_left, " ", E_right)
            end
            pars.verb > 0 && println(results[end])
        end
    end
    return results
end


function findSigma0!(   op, ep, pars;
                        tol = 1e-4, dm = 1, smallsteps = false, maxstep= 10.5,
                        fixρ=true, fixnorm=false, targetΣ = 0., maximum_m=-1, testρ=false
                        )
    mlist = Any[]
    Σlist = Any[]

    if maximum_m>0 && op.m > maximum_m
        return false
    end
    ###PRIMO TENTATIVO
    println("@@@ T 1 : m=$(op.m)")

    converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm, testρ=testρ)
    if testρ && ep.ρ<1e-5
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

        converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm, testρ=testρ)
        if testρ && ep.ρ<1e-5
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
        converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm, testρ=testρ)
        if testρ && ep.ρ<1e-5
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


function findmaximumSigma!(   op, ep, pars;
                        tol = 1e-9, dm = 1,
                        fixρ=true, fixnorm=false, δ=1e-2
                        )
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

    return true
end

# function findSigma0!(op, ep, pars;
#                 tol=1e-4, dm=10, smallsteps=false, fixnorm=false, fixρ=true)
#     mlist = Any[]
#     Σlist = Any[]
#
#     ###PRIMO TENTATIVO
#     println("@@@ T 1 : m=$(op.m)")
#     ok = converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm)
#     tf = all_therm_func(op, ep)
#     println(tf)
#     push!(mlist, op.m)
#     push!(Σlist, tf.Σ)
#     absSigma = abs(tf.Σ)
#
#     println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")
#     ###SECOND TENTATIVO
#     if absSigma > tol
#         op.m += abs(op.m * tf.Σ * dm) > 0.5 ? 0.5*sign(op.m * tf.Σ * dm) : op.m * tf.Σ * dm
#         println("@@@ T 2 : m=$(op.m)")
#
#         ok = converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm)
#         tf = all_therm_func(op, ep)
#         println(tf)
#         push!(mlist, op.m)
#         push!(Σlist, tf.Σ)
#         absSigma = abs(tf.Σ)
#         println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")
#     end
#
#     ###ALTRI  TENTATIVI
#     trial = 3
#     while absSigma > tol
#         s = 0
#         if trial >= 3
#             s = -(mlist[end]*Σlist[end-1] - mlist[end-1]*Σlist[end])/(Σlist[end]-Σlist[end-1])
#         end
#         if smallsteps && abs(s - op.m) >  op.m * abs(tf.Σ) * dm
#             op.m += sign(s - op.m) * min(op.m * abs(tf.Σ) * dm, 0.5)
#         else
#             op.m = s
#         end
#         println("@@@ T $(trial) : m=$(op.m)")
#         ok = converge!(op, ep, pars, fixm=true, fixρ=fixρ, fixnorm=fixnorm)
#
#         tf = all_therm_func(op, ep)
#         println(tf)
#         println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")
#         push!(mlist, op.m)
#         push!(Σlist, tf.Σ)
#         absSigma = abs(tf.Σ)
#         trial += 1
#     end
#
#     return ok
# end

function readparams(file; line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    ep = ExtParams(res[line,1:3]...)
    op = OrderParams(res[line,7:end]...)
    tf = ThermFunc(res[line,4:6]...)
    return ep, op, tf
end

end ## module
