module PhaseRetr

# using LittleScienceTools.Roots
using QuadGK
using AutoGrad
using Cubature
using FastGaussQuadrature
# import LsqFit: curve_fit
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 15.0
const dx = 0.05

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., atol=1e-7, maxevals=5*10^3)[1]


let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    global gw
    gw(n::Int) = get!(s, n) do
        (x,w) = gausshermite(n)
        return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
    end
end

function ∫DD(f; n=301)
    (xs, ws) = gw(n)
    s = 0.0
    for (x,w) in zip(xs, ws)
        y = f(x)
        s += w  * ifelse(isfinite(y), y, 0.0)
    end

    return s
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
    xp1 = deepcopy(x)
    xm1 = deepcopy(x)
    setfield!(xp1[arg], i, getfield(xp1[arg], i) + δ)
    setfield!(xm1[arg], i, getfield(xm1[arg], i) - δ)
    fp1 = f(xp1...)
    fm1 = f(xm1...)
    return (fp1-fm1) / (2δ)
end


############### PARAMS ################

mutable struct OrderParams
    q0::Float64 # eventually q0=1
    δq::Float64
    qh0::Float64
    δqh::Float64
    ρh::Float64
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
    λ::Float64
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
function Gi(op, ep)
    @extract op: q0 δq qh0 δqh ρh
    @extract ep: ρ
    (δqh*q0 - 2*ρ*ρh - δq*qh0)/2 #CHECK
end

#### ENTROPIC TERM ####

Gs(op, ep) = 0.5*(Power(op.ρh,2) + op.qh0)/(op.δqh + ep.λ)

#### ENERGETIC TERM ####


function argGe(y, h, δq)
    ### find min of 1/2 u^2 + (y - abs(√δq*u + h))^2
    # u* = 2√δq*(y-|z0|)*sign(h) / (1+2δq) : -h/√δq
    abs(h) > - 2δq*y ? (y - abs(h))^2 / (1 + 2δq) : h^2/(2δq) + y^2
end

function ∂δq_argGe(y, h, δq)
    ### find min of 1/2 u^2 + (y - abs(√δq*u + h))^2
    abs(h) > - 2δq*y ? -2*(y - abs(h))^2 / (1 + 2δq)^2 : -h^2/(2*δq^2)
end

function ∂y_argGe(y, h, δq)
    ### find min of 1/2 u^2 + (y - abs(√δq*u + h))^2
    abs(h) > - 2δq*y ? 2*(y - abs(h)) / (1 + 2δq) : 2*y
end

function ∂h_argGe(y, h, δq)
    ### find min of 1/2 u^2 + (y - abs(√δq*u + h))^2
    abs(h) > - 2δq*y ? -sign(h)*2*(y - abs(h)) / (1 + 2δq) : 2*h/(2δq)
end

function f1(a,b,c,d,g,z0)
    (b+d)*(2*(a+c)-(b+d)*g)*z0*G(-z0*g) + ((b+d)^2+(a+c)^2*z0^2)*H(-z0*g)
end
function df1_1_3(a,b,c,d,g,z0)
    (b+d)*2*z0*G(-z0*g) + 2*(a+c)*z0^2*H(-z0*g)
end
function df1_2_4(a,b,c,d,g,z0)
    2*((a+c)-(b+d)*g)*z0*G(-z0*g) + 2*(b+d)*H(-z0*g)
end
function df1_5(a,b,c,d,g,z0)
    -(b+d)*(2*(a+c)-(b+d)*g)*z0^3*g*G(-z0*g) + (a+c)^2*z0^2*G(-z0*g)*z0
end



function f2(a,b,c,d,g,δq,z0)
    (-(2δq*a*b+c*d)-(2δq*b^2+d^2)/2*g)*z0*G(-z0*g) + ((2δq*b^2+d^2)/2 + (2δq*a^2+c^2)*z0^2/2)*H(-z0*g)
end
function df2_1(a,b,c,d,g,δq,z0)
    -2δq*b*z0*G(-z0*g) + 2δq*a*z0^2*H(-z0*g)
end
function df2_2(a,b,c,d,g,δq,z0)
    -2δq*(a+b*g)*z0*G(-z0*g) + 2δq*b*H(-z0*g)
end
function df2_3(a,b,c,d,g,δq,z0)
    -d*z0*G(-z0*g) + c*z0^2*H(-z0*g)
end
function df2_4(a,b,c,d,g,δq,z0)
    (-c-d*g)*z0*G(-z0*g) + d*H(-z0*g)
end
function df2_5(a,b,c,d,g,δq,z0)
    ((2δq*a*b+c*d)+(2δq*b^2+d^2)/2*g)*z0^3*g*G(-z0*g) + ((2δq*a^2+c^2)*z0^2/2)*G(-z0*g)*z0
end
function df2_6(a,b,c,d,g,δq,z0)
    (-2*a*b-b^2*g)*z0*G(-z0*g)+(b^2+a^2*z0^2)*H(-z0*g)
end

function Ge(op, ep)
    @extract op: q0 δq
    @extract ep: ρ Δ
    ### noiseless
    # -∫D(u0->∫D(z0->begin
    #     h = ρ*u0 + √(q0 - ρ^2)*z0
    #     argGe(abs(u0), h, δq)
    # end))
    ### noisy
    # return -∫D(z0->∫D(u0->∫D(y->begin
    #     y0 = abs(√(1-ρ^2/q0)*u0 + ρ/√q0*z0)
    #     argGe(√Δ*y + y0, √q0*z0, δq)
    # end)))
    ### noisy (integrate u0)
    # a1 = √(Δ+(1-ρ^2/q0))
    # b1 = (ρ/√q0)
    # c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
    # d1 = (√(1-ρ^2/q0)/√Δ)
    # ge = -2∫D(z0->∫D(y->begin
    #         argGe(a1*y + b1*z0, √q0*z0, δq) * H(-(c1*z0 + d1*y))
    #     end))
    # return ge
    if Δ == 0
        ge = (-(π*(1 + q0)) + 4*√(q0 - ρ^2) + 4*ρ*atan(ρ/√(q0 - ρ^2)))/(π*(1+2*δq))
    else

        a = √(q0-ρ^2)
        b = √(1 + Δ*q0/a^2)
        # da_q0 = 1/(2a)
        # da_ρ = -ρ/a
        # db_q0 = Δ/(2b*a^2)*(1-2*q0/a*da_q0)
        # db_ρ = -1/b*Δ*q0/a^3*da_ρ

        x1 = b
        # dx1_q0 = db_q0
        # dx1_ρ = db_ρ

        x2 = -ρ*Δ/a
        # dx2_q0 = ρ*Δ/a^2*da_q0
        # dx2_ρ = -Δ/a*(1-ρ/a*da_ρ)

        x3 = ρ*b
        # dx3_q0 = ρ*db_q0
        # dx3_ρ = b+ρ*db_ρ

        x4 = a
        # dx4_q0 = da_q0
        # dx4_ρ = da_ρ

        ge = -2∫D(z0-> begin
                if 2δq*x2 + x4 > 0
                    if z0 > 0
                        (f1(x1,x2,-x3,-x4,x3/x4,z0)+f1(-x1,x2,-x3,x4,-x3/x4,z0))/(1+2δq)
                    else
                        (f1(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+f1(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0))/(1+2δq) +
                        (f2(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)-f2(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0))/δq
                    end
                else
                    if z0 > 0
                        (f1(x1,x2,-x3,-x4,x3/x4,z0)+f1(-x1,x2,-x3,x4,-x3/x4,z0)-f1(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0))/(1+2δq) +
                        f2(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)/δq
                    else
                        f1(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)/(1+2δq) + f2(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)/δq
                    end
                end * H(-(√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0)) / ((1+Δ*ρ^2/(q0-ρ^2)))
            end)
    end
    return ge
end

function ∂q0_Ge(op, ep)
    @extract op: q0 δq
    @extract ep: ρ Δ
    if Δ == 0
        dq = -((q0 - 2/π*√(q0 - ρ^2))/(q0*(1+2*δq)))
    else
        # a1 = √(Δ+(1-ρ^2/q0))
        # da1 = 1/2*ρ^2/(q0^2*√(Δ+(1-ρ^2/q0)))
        # b1 = (ρ/√q0)
        # db1 = -1/2*ρ/(q0)^(3/2)
        # c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
        # dc1 = -1/2*ρ/(q0)^(3/2)*√(1/(1-ρ^2/q0)+ 1/Δ) + 1/2*ρ^(3)/√q0/√(1/(1-ρ^2/q0)+ 1/Δ)*1/(1-ρ^2/q0)^2/q0^2
        # d1 = (√(1-ρ^2/q0)/√Δ)
        # dd1 = 1/2*ρ^2/(q0^2*√(1-ρ^2/q0)*√Δ)
        # dq = -2∫D(z0->∫D(y->begin
        #         ∂y_argGe(a1*y + b1*z0, √q0*z0, δq) * (da1*y + db1*z0) * H(-(c1*z0 + d1*y)) +
        #         ∂h_argGe(a1*y + b1*z0, √q0*z0, δq) * (z0/(2*√q0)) * H(-(c1*z0 + d1*y)) +
        #         argGe(a1*y + b1*z0, √q0*z0, δq) * G(-(c1*z0 + d1*y)) * ( dc1*z0 + dd1*y)
        # end))
        # return dq

        a = √(q0-ρ^2)
        da_q0 = 1/(2a)
        b = √(1 + Δ*q0/a^2)
        db_q0 = Δ/(2b*a^2)*(1-2*q0/a*da_q0)

        x1 = b
        dx1_q0 = db_q0

        x2 = -ρ*Δ/a
        dx2_q0 = ρ*Δ/a^2*da_q0

        x3 = ρ*b
        dx3_q0 = ρ*db_q0

        x4 = a
        dx4_q0 = da_q0

        dq = -2∫D(z0-> begin
            if 2δq*x2 + x4 > 0
                if z0 > 0
                    (dx1_q0 * df1_1_3(x1,x2,-x3,-x4,x3/x4,z0) +
                    dx2_q0 * df1_2_4(x1,x2,-x3,-x4,x3/x4,z0) +
                    dx3_q0 * (1/x4*df1_5(x1,x2,-x3,-x4,x3/x4,z0) - df1_1_3(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx4_q0 * (-x3/x4^2*df1_5(x1,x2,-x3,-x4,x3/x4,z0)-df1_2_4(x1,x2,-x3,-x4,x3/x4,z0)) +
                    -dx1_q0 * df1_1_3(-x1,x2,-x3,x4,-x3/x4,z0) +
                    dx2_q0 * df1_2_4(-x1,x2,-x3,x4,-x3/x4,z0) +
                    dx3_q0 * (-1/x4*df1_5(-x1,x2,-x3,x4,-x3/x4,z0) -df1_1_3(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    dx4_q0 * (x3/x4^2*df1_5(-x1,x2,-x3,x4,-x3/x4,z0) +df1_2_4(-x1,x2,-x3,x4,-x3/x4,z0))) /(1+2δq)
                else
                    (
                    dx1_q0 * (2δq/(2δq*x2+x4)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+df1_1_3(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx2_q0 * (-2δq*(2δq*x1+x3)/(2δq*x2+x4)^2*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+df1_2_4(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx3_q0 * (1/(2δq*x2+x4)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)-df1_1_3(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx4_q0 * (-(2δq*x1+x3)/(2δq*x2+x4)^2*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)-df1_2_4(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0))+
                    dx1_q0 * (-2δq/(2δq*x2-x4)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)-df1_1_3(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx2_q0 * (2δq*(2δq*x1-x3)/(2δq*x2-x4)^2*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)+df1_2_4(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx3_q0 * (1/(2δq*x2-x4)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)-df1_1_3(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx4_q0 * (-(2δq*x1-x3)/(2δq*x2-x4)^2*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)+df1_2_4(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0))
                    )/(1+2δq) +
                    (
                    dx1_q0 * (-2δq/(2δq*x2+x4)*df2_5(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_1(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx2_q0 * (2δq*(2δq*x1+x3)/(2δq*x2+x4)^2*df2_5(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_2(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx3_q0 * (-1/(2δq*x2+x4)*df2_5(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_3(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx4_q0 * ((2δq*x1+x3)/(2δq*x2+x4)^2*df2_5(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_4(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    -(
                    dx1_q0 * (-2δq/(2δq*x2-x4)*df2_5(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_1(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx2_q0 * (2δq*(2δq*x1-x3)/(2δq*x2-x4)^2*df2_5(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_2(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx3_q0 * (1/(2δq*x2-x4)*df2_5(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_3(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx4_q0 * (-(2δq*x1-x3)/(2δq*x2-x4)^2*df2_5(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_4(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0))
                    ))/δq
                end
            else
                if z0 > 0
                    (
                    dx1_q0 * (df1_1_3(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx2_q0 * (df1_2_4(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx3_q0 * (1/x4*df1_5(x1,x2,-x3,-x4,x3/x4,z0)-df1_1_3(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx4_q0 * (-x3/x4^2*df1_5(x1,x2,-x3,-x4,x3/x4,z0)-df1_2_4(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx1_q0 * (-df1_1_3(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    dx2_q0 * (df1_2_4(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    dx3_q0 * (-1/x4*df1_5(-x1,x2,-x3,x4,-x3/x4,z0)-df1_1_3(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    dx4_q0 * (x3/x4^2*df1_5(-x1,x2,-x3,x4,-x3/x4,z0)+df1_2_4(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    -(dx1_q0 * (2δq/(2δq*x2+x4)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+df1_1_3(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx2_q0 * (-2δq*(2δq*x1+x3)/(2δq*x2+x4)^2*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+df1_2_4(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx3_q0 * (1/(2δq*x2+x4)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)-df1_1_3(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx4_q0 * (-(2δq*x1+x3)/(2δq*x2+x4)^2*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)-df1_2_4(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0))
                    ))/(1+2δq) +
                    (dx1_q0 * (2δq/(2δq*x2+x4)*df2_5(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)-df2_1(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx2_q0 * (-2δq*(2δq*x1+x3)/(2δq*x2+x4)^2*df2_5(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_2(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx3_q0 * (1/(2δq*x2+x4)*df2_5(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)-df2_3(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx4_q0 * (-(2δq*x1+x3)/(2δq*x2+x4)^2*df2_5(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_4(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0))
                    )/δq
                else
                    (dx1_q0 * (-2δq/(2δq*x2-x4)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)-df1_1_3(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx2_q0 * (2δq*(2δq*x1-x3)/(2δq*x2-x4)^2*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)+df1_2_4(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx3_q0 * (1/(2δq*x2-x4)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)-df1_1_3(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx4_q0 * (-(2δq*x1-x3)/(2δq*x2-x4)^2*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)+df1_2_4(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0))
                    )/(1+2δq) +
                    (dx1_q0 * (2δq/(2δq*x2-x4)*df2_5(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)-df2_1(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx2_q0 * (-2δq*(2δq*x1-x3)/(2δq*x2-x4)^2*df2_5(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_2(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx3_q0 * (-1/(2δq*x2-x4)*df2_5(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)-df2_3(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx4_q0 * ((2δq*x1-x3)/(2δq*x2-x4)^2*df2_5(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_4(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0))
                    )/δq
                end
            end * H(-(√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0)) / (1+Δ*ρ^2/(q0-ρ^2)) +
            if 2δq*x2 + x4 > 0
                if z0 > 0
                    (f1(x1,x2,-x3,-x4,x3/x4,z0)+f1(-x1,x2,-x3,x4,-x3/x4,z0))/(1+2δq)
                else
                    (f1(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+f1(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0))/(1+2δq) +
                    (f2(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)-f2(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0))/δq
                end
            else
                if z0 > 0
                    (f1(x1,x2,-x3,-x4,x3/x4,z0)+f1(-x1,x2,-x3,x4,-x3/x4,z0)-f1(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0))/(1+2δq) +
                    f2(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)/δq
                else
                    f1(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)/(1+2δq) + f2(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)/δq
                end
            end * ( H(-(√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0)) / (1+Δ*ρ^2/(q0-ρ^2))^2 * Δ*ρ^2/(q0-ρ^2)^2 +
                    -1/2 * G(-(√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0)) / (1+Δ*ρ^2/(q0-ρ^2)) * (1/√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0*Δ*ρ^2/(q0-ρ^2)^2))
        end)
        # op.q0 += 1e-7
        # ge1 = Ge(op,ep)
        # op.q0 -= 2e-7
        # ge2 = Ge(op,ep)
        # op.q0 += 1e-7
        # dq = (ge1 - ge2) / 2e-7
    end
    return dq
end

function ∂δq_Ge(op, ep)
    @extract op: q0 δq
    @extract ep: ρ Δ
    if Δ == 0
        ddq = (2*(π + π*q0 - 4*√(q0 - ρ^2) - 4*ρ*atan(ρ/√(q0 - ρ^2))))/((1 + 2*δq)^2*π)
    else
        # a1 = √(Δ+(1-ρ^2/q0))
        # b1 = (ρ/√q0)
        # c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
        # d1 = (√(1-ρ^2/q0)/√Δ)
        # ddq = -∫D(z0->∫D(y->begin
        #         ∂δq_argGe(a1*y + b1*z0, √q0*z0, δq) * H(-(c1*z0 + d1*y)) +
        #         ∂δq_argGe(a1*y - b1*z0, √q0*z0, δq) * H( (c1*z0 - d1*y))
        #     end))
        # return ddq

        a = √(q0-ρ^2)
        b = √(1 + Δ*q0/a^2)
        x1 = b
        x2 = -ρ*Δ/a
        x3 = ρ*b
        x4 = a

        ddq = -2∫D(z0-> begin
        if 2δq*x2 + x4 > 0
            if z0 > 0
                0
            else
                (
                (2x1/(2δq*x2+x4)-2*x2*(2δq*x1+x3)/(2δq*x2+x4)^2)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0) +
                (-2x1/(2δq*x2-x4)+2*x2*(2δq*x1-x3)/(2δq*x2-x4)^2)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)
                )/(1+2δq) +
                (
                (-2x1/(2δq*x2+x4)+2*x2*(2δq*x1+x3)/(2δq*x2+x4)^2)*df2_5(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_6(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0) +
                -(
                (-2x1/(2δq*x2-x4)+2*x2*(2δq*x1-x3)/(2δq*x2-x4)^2)*df2_5(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_6(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)
                ))/δq
            end
        else
            if z0 > 0
                (
                (2x1/(2δq*x2+x4)-2*x2*(2δq*x1+x3)/(2δq*x2+x4)^2)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)
                )/(1+2δq) +
                (
                (2x1/(2δq*x2+x4)-2x2*(2δq*x1+x3)/(2δq*x2+x4)^2)*df2_5(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_6(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0))/δq
            else
                ((-2x1/(2δq*x2-x4)+2x2*(2δq*x1-x3)/(2δq*x2-x4)^2)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)
                )/(1+2δq) +
                ((2x1/(2δq*x2-x4)-2x2*(2δq*x1-x3)/(2δq*x2-x4)^2)*df2_5(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_6(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0))/δq
            end
        end * H(-(√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0)) / (1+Δ*ρ^2/(q0-ρ^2)) +
        if 2δq*x2 + x4 > 0
            if z0 > 0
                (f1(x1,x2,-x3,-x4,x3/x4,z0)+f1(-x1,x2,-x3,x4,-x3/x4,z0))*2/(1+2δq)^2
            else
                (f1(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+f1(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0))*2/(1+2δq)^2 +
                (f2(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)-f2(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0))/δq^2
            end
        else
            if z0 > 0
                (f1(x1,x2,-x3,-x4,x3/x4,z0)+f1(-x1,x2,-x3,x4,-x3/x4,z0)-f1(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0))*2/(1+2δq)^2 +
                f2(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)/δq^2
            else
                f1(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)*2/(1+2δq)^2 + f2(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)/δq^2
            end
        end * ( -H(-(√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0)) / (1+Δ*ρ^2/(q0-ρ^2)))
    end)
        # op.δq += 1e-7
        # ge1 = Ge(op,ep)
        # op.δq -= 2e-7
        # ge2 = Ge(op,ep)
        # op.δq += 1e-7
        # ddq = (ge1 - ge2) / 2e-7
    end
    return ddq
end

function ∂ρ_Ge(op, ep)
    @extract op: q0 δq
    @extract ep: ρ Δ
    if Δ == 0
        dρ = (4*atan(ρ/√(q0 - ρ^2)))/(π + 2*δq*π)
    else
        # a1 = √(Δ+(1-ρ^2/q0))
        # da1 = -ρ/(q0*√(Δ+(1-ρ^2/q0)))
        # b1 = (ρ/√q0)
        # db1 = 1/√q0
        # c1 = ρ/√q0*√(1/(1-ρ^2/q0)+ 1/Δ)
        # dc1 = 1/√q0*√(1/(1-ρ^2/q0)+ 1/Δ) + ρ/√q0/√(1/(1-ρ^2/q0)+ 1/Δ)*ρ/q0/(1-ρ^2/q0)^2
        # d1 = √(1-ρ^2/q0)/√Δ
        # dd1 = -ρ/(q0*√(1-ρ^2/q0)*√Δ)
        # dρ = -∫D(z0->∫D(y->begin
        #         ∂y_argGe(a1*y + b1*z0, √q0*z0, δq) * (da1*y + db1*z0) * H(-(c1*z0 + d1*y)) +
        #         ∂y_argGe(a1*y - b1*z0, √q0*z0, δq) * (da1*y - db1*z0) * H( (c1*z0 - d1*y)) +
        #         argGe(a1*y + b1*z0, √q0*z0, δq) * G(-(c1*z0 + d1*y)) * ( dc1*z0 + dd1*y)  +
        #         argGe(a1*y - b1*z0, √q0*z0, δq) * G( (c1*z0 - d1*y)) * (-dc1*z0 + dd1*y)
        #     end))
        # return dρ

        a = √(q0-ρ^2)
        b = √(1 + Δ*q0/a^2)
        da_ρ = -ρ/a
        db_ρ = -1/b*Δ*q0/a^3*da_ρ

        x1 = b
        dx1_ρ = db_ρ

        x2 = -ρ*Δ/a
        dx2_ρ = -Δ/a*(1-ρ/a*da_ρ)

        x3 = ρ*b
        dx3_ρ = b+ρ*db_ρ

        x4 = a
        dx4_ρ = da_ρ

        dρ = -2∫D(z0-> begin
            if 2δq*x2 + x4 > 0
                if z0 > 0
                    (dx1_ρ * df1_1_3(x1,x2,-x3,-x4,x3/x4,z0) +
                    dx2_ρ * df1_2_4(x1,x2,-x3,-x4,x3/x4,z0) +
                    dx3_ρ * (1/x4*df1_5(x1,x2,-x3,-x4,x3/x4,z0) - df1_1_3(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx4_ρ * (-x3/x4^2*df1_5(x1,x2,-x3,-x4,x3/x4,z0)-df1_2_4(x1,x2,-x3,-x4,x3/x4,z0)) +
                    -dx1_ρ * df1_1_3(-x1,x2,-x3,x4,-x3/x4,z0) +
                    dx2_ρ * df1_2_4(-x1,x2,-x3,x4,-x3/x4,z0) +
                    dx3_ρ * (-1/x4*df1_5(-x1,x2,-x3,x4,-x3/x4,z0) -df1_1_3(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    dx4_ρ * (x3/x4^2*df1_5(-x1,x2,-x3,x4,-x3/x4,z0) +df1_2_4(-x1,x2,-x3,x4,-x3/x4,z0))) /(1+2δq)
                else
                    (
                    dx1_ρ * (2δq/(2δq*x2+x4)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+df1_1_3(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx2_ρ * (-2δq*(2δq*x1+x3)/(2δq*x2+x4)^2*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+df1_2_4(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx3_ρ * (1/(2δq*x2+x4)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)-df1_1_3(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx4_ρ * (-(2δq*x1+x3)/(2δq*x2+x4)^2*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)-df1_2_4(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0))+
                    dx1_ρ * (-2δq/(2δq*x2-x4)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)-df1_1_3(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx2_ρ * (2δq*(2δq*x1-x3)/(2δq*x2-x4)^2*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)+df1_2_4(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx3_ρ * (1/(2δq*x2-x4)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)-df1_1_3(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx4_ρ * (-(2δq*x1-x3)/(2δq*x2-x4)^2*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)+df1_2_4(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0))
                    )/(1+2δq) +
                    (
                    dx1_ρ * (-2δq/(2δq*x2+x4)*df2_5(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_1(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx2_ρ * (2δq*(2δq*x1+x3)/(2δq*x2+x4)^2*df2_5(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_2(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx3_ρ * (-1/(2δq*x2+x4)*df2_5(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_3(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx4_ρ * ((2δq*x1+x3)/(2δq*x2+x4)^2*df2_5(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_4(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    -(
                    dx1_ρ * (-2δq/(2δq*x2-x4)*df2_5(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_1(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx2_ρ * (2δq*(2δq*x1-x3)/(2δq*x2-x4)^2*df2_5(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_2(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx3_ρ * (1/(2δq*x2-x4)*df2_5(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_3(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx4_ρ * (-(2δq*x1-x3)/(2δq*x2-x4)^2*df2_5(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_4(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0))
                    ))/δq
                end
            else
                if z0 > 0
                    (
                    dx1_ρ * (df1_1_3(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx2_ρ * (df1_2_4(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx3_ρ * (1/x4*df1_5(x1,x2,-x3,-x4,x3/x4,z0)-df1_1_3(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx4_ρ * (-x3/x4^2*df1_5(x1,x2,-x3,-x4,x3/x4,z0)-df1_2_4(x1,x2,-x3,-x4,x3/x4,z0)) +
                    dx1_ρ * (-df1_1_3(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    dx2_ρ * (df1_2_4(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    dx3_ρ * (-1/x4*df1_5(-x1,x2,-x3,x4,-x3/x4,z0)-df1_1_3(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    dx4_ρ * (x3/x4^2*df1_5(-x1,x2,-x3,x4,-x3/x4,z0)+df1_2_4(-x1,x2,-x3,x4,-x3/x4,z0)) +
                    -(dx1_ρ * (2δq/(2δq*x2+x4)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+df1_1_3(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx2_ρ * (-2δq*(2δq*x1+x3)/(2δq*x2+x4)^2*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+df1_2_4(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx3_ρ * (1/(2δq*x2+x4)*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)-df1_1_3(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)) +
                    dx4_ρ * (-(2δq*x1+x3)/(2δq*x2+x4)^2*df1_5(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)-df1_2_4(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0))
                    ))/(1+2δq) +
                    (dx1_ρ * (2δq/(2δq*x2+x4)*df2_5(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)-df2_1(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx2_ρ * (-2δq*(2δq*x1+x3)/(2δq*x2+x4)^2*df2_5(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_2(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx3_ρ * (1/(2δq*x2+x4)*df2_5(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)-df2_3(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)) +
                    dx4_ρ * (-(2δq*x1+x3)/(2δq*x2+x4)^2*df2_5(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)+df2_4(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0))
                    )/δq
                else
                    (dx1_ρ * (-2δq/(2δq*x2-x4)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)-df1_1_3(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx2_ρ * (2δq*(2δq*x1-x3)/(2δq*x2-x4)^2*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)+df1_2_4(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx3_ρ * (1/(2δq*x2-x4)*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)-df1_1_3(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)) +
                    dx4_ρ * (-(2δq*x1-x3)/(2δq*x2-x4)^2*df1_5(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)+df1_2_4(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0))
                    )/(1+2δq) +
                    (dx1_ρ * (2δq/(2δq*x2-x4)*df2_5(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)-df2_1(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx2_ρ * (-2δq*(2δq*x1-x3)/(2δq*x2-x4)^2*df2_5(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_2(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx3_ρ * (-1/(2δq*x2-x4)*df2_5(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)-df2_3(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)) +
                    dx4_ρ * ((2δq*x1-x3)/(2δq*x2-x4)^2*df2_5(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)+df2_4(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0))
                    )/δq
                end
            end * H(-(√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0)) / (1+Δ*ρ^2/(q0-ρ^2)) +
            if 2δq*x2 + x4 > 0
                if z0 > 0
                    (f1(x1,x2,-x3,-x4,x3/x4,z0)+f1(-x1,x2,-x3,x4,-x3/x4,z0))/(1+2δq)
                else
                    (f1(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0)+f1(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0))/(1+2δq) +
                    (f2(x1,x2,x3,x4,-(2δq*x1+x3)/(2δq*x2+x4),δq,z0)-f2(x1,x2,x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),δq,z0))/δq
                end
            else
                if z0 > 0
                    (f1(x1,x2,-x3,-x4,x3/x4,z0)+f1(-x1,x2,-x3,x4,-x3/x4,z0)-f1(x1,x2,-x3,-x4,(2δq*x1+x3)/(2δq*x2+x4),z0))/(1+2δq) +
                    f2(-x1,x2,-x3,x4,(2δq*x1+x3)/(2δq*x2+x4),δq,z0)/δq
                else
                    f1(-x1,x2,-x3,x4,-(2δq*x1-x3)/(2δq*x2-x4),z0)/(1+2δq) + f2(-x1,x2,-x3,x4,(2δq*x1-x3)/(2δq*x2-x4),δq,z0)/δq
                end
            end * ( -H(-(√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0)) / (1+Δ*ρ^2/(q0-ρ^2))^2 * (2Δ*ρ/(q0-ρ^2) + 2*Δ*ρ^3/(q0-ρ^2)^2)+
                    1/2 * G(-(√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0)) / (1+Δ*ρ^2/(q0-ρ^2)) * (1/√((1+Δ*ρ^2/(q0-ρ^2)))/√Δ*z0*(2Δ*ρ/(q0-ρ^2) + 2*Δ*ρ^3/(q0-ρ^2)^2)))
        end)
        # ep.ρ += 1e-7
        # ge1 = Ge(op,ep)
        # ep.ρ -= 2e-7
        # ge2 = Ge(op,ep)
        # ep.ρ += 1e-7
        # dρ = (ge1 - ge2) / 2e-7

    end
    return dρ
end

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    Gi(op, ep) + Gs(op, ep) + ep.α*Ge(op, ep)
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
fδqh(op, ep) = -2ep.α * ∂q0_Ge(op, ep)
fqh0(op, ep) = 2ep.α * ∂δq_Ge(op, ep)
fρh(op, ep) = ep.α * ∂ρ_Ge(op, ep)

fq0(op, ep) = (op.ρh^2 + op.qh0) / (op.δqh + ep.λ)^2
fδq(op, ep) = 1 / (op.δqh + ep.λ)
fρ(op, ep) = op.ρh / (op.δqh + ep.λ)

iρh(op, ep) = (true, ep.ρ*(op.δqh + ep.λ))

function iδqh(op)
    (true, sqrt(op.qh0 + op.ρh^2))
end

###############################

function fix_inequalities!(op, ep)
    if  op.q0 < ep.ρ^2 + 1e-8
        op.q0 = ep.ρ^2 + 1e-8
    end
    if op.δq < 0
        op.δq = rand()
    end
end

function converge!(op::OrderParams, ep::ExtParams, pars::Params;
        fixρ=true, fixnorm=true, extrap=-1, testρ=false)
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
            @updateI op.ρh oki   iρh   Δ 0. verb  op ep
            ok &= oki
        else
            @update  op.ρh  fρh       Δ ψ verb  op ep
        end

        @update op.δq   fδq       Δ ψ verb  op ep
        if !fixnorm
            @update op.q0   fq0     Δ ψ verb  op ep
        end
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op ep
        end

        fix_inequalities!(op, ep)

        verb > 1 && println(" Δ=$Δ\n")
        verb > 2 && it%5==0 && (println(ep);println(all_therm_func(op, ep));println(op))

        @assert isfinite(Δ)
        ok &= Δ < ϵ
        if ok || (testρ && ep.ρ>0.3)
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
        q0 = 1.,
        δq=0.5,
        qh0=0., δqh=0.6,
        ρ=0, ρh=0,
        α=0.1,Δ=0.,λ=0.,
        ϵ=1e-4, maxiters=100000, verb=3, ψ=0.,
        fixρ=true, fixnorm=true, extrap=-1
    )
    op = OrderParams(q0,δq,qh0,δqh,ρh)
    ep = ExtParams(α, ρ, Δ, λ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixρ=fixρ, fixnorm=fixnorm,extrap=extrap)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
        q0 = 1,
        δq=0.3188,
        qh0=0.36889,δqh=0.36889, ρh=0.56421,
        ρ=0.384312, α=1, Δ=0., λ=0.,
        ϵ=1e-4, maxiters=10000,verb=3, ψ=0.,
        kws...)

    op = OrderParams(q0,δq,qh0,δqh,ρh)
    ep = ExtParams(first(α), first(ρ), Δ, λ)
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; ρ=ρ,α=α, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
    α=1, ρ=1,
    resfile = "results.txt",
    fixρ=true, fixnorm=false, extrap=-1)

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

function find_best_start_params(resfile, α, Δ, λ, left)
    res = readdlm(resfile)
    resmap = map(i->res[i,:],1:size(res,1))
    left ? filter!(i->i[end]=="left",resmap) : filter!(i->i[end]=="right",resmap)
    c = map(i->abs(resmap[i][1]-α) + abs(resmap[i][3]-Δ) + abs(resmap[i][4]), 1:length(resmap))
    sp1 = sortperm(c)[1]
    ep = ExtParams(resmap[sp1][1:4]...)
    op = OrderParams(resmap[sp1][6:end-1]...)
    tf = ThermFunc(resmap[sp1][5]...)
    return ep, op, tf
end

function spanΔ_λ(;
        q0 = 1,
        δq=0.3188,
        qh0=0.36889,δqh=0.36889, ρh=0.56421,
        ρ=0., α=1, Δ=1., λ=0,
        ϵ=1e-4, maxiters=10000,verb=3, ψ=0.,
        kws...)

    op = OrderParams(q0,δq,qh0,δqh,ρh)
    ep = ExtParams(first(α), ρ, first(Δ), first(λ))
    pars = Params(ϵ, ψ, maxiters, verb)
    return spanΔ_λ!(op, ep, pars; Δs=Δ, αs=α, λs=λ, kws...)
end

function spanΔ_λ!(op::OrderParams, ep::ExtParams, pars::Params;
    αs=1, Δs=1, λs=1,
    resfile = "results.txt",
    resfile2 = "results2.txt",
    fixρ=true, fixnorm=false, extrap=-1)

    if !isfile(resfile2)
        f = open(resfile2, "w")
        println(f, "### 1:α 2:Δ 3:λ 4:ρ_left 5:ρ_right 6:E_left 7:E_right")
        close(f)
    end

    ϵ = pars.ϵ
    maxiters = pars.maxiters

    op1 = deepcopy(op)
    ep1 = deepcopy(ep)

    results = []
    for α in αs, Δ in Δs, λ in λs
            ep.α = α;
            ρ_left = -1
            E_left = -1
            ρ_right = -1
            E_right = -1

            ep, op, _ = find_best_start_params(resfile, α, Δ, λ, true)
            ep.α = α;
            ep.Δ = Δ;
            ep.λ = λ;
            already_done = false

            pars.verb > 0 && println(" >>>>>", ep)
            done = false
            while !done
                ep.ρ = 0.0;
                try
                    pars.ϵ = 1e-30;
                    pars.maxiters = 10
                    ok = converge!(op, ep, pars; fixρ=true,fixnorm=fixnorm)
                    pars.ϵ = ϵ;
                    pars.maxiters = maxiters
                    ok = converge!(op, ep, pars; fixρ=true,fixnorm=fixnorm)
                catch
                    break
                end
                tf = all_therm_func(op, ep)
                open(resfile, "a") do rf
                    println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op), " left")
                end
                E1 = -tf.ϕ

                ep.ρ = 1e-5
                pars.ϵ = 1e-30;
                pars.maxiters = 30
                ok = converge!(op, ep, pars; fixρ=false, testρ=true, fixnorm=fixnorm)
                pars.ϵ = ϵ;
                pars.maxiters = maxiters
                if ep.ρ < 1e-5
                    ρ_left = 0
                    E_left = E1
                else
                    ρ_left = 10
                    E_left = -1
                end
                done = true
            end

            ep, op, _ = find_best_start_params(resfile, α, Δ, λ, false)
            ep.α = α;
            ep.Δ = Δ;
            ep.λ = λ;

            done= false
            while !done
                try
                    pars.ϵ = 1e-30;
                    pars.maxiters = 10
                    ok = converge!(op, ep, pars; fixρ=true,fixnorm=fixnorm)
                    pars.ϵ = ϵ;
                    pars.maxiters = maxiters
                    ok = converge!(op, ep, pars; fixρ=true,fixnorm=fixnorm)
                catch
                    break
                end
                op1 = deepcopy(op)
                ep1 = deepcopy(op)
                tf1 = all_therm_func(op, ep)

                pars.ϵ = 1e-30;
                pars.maxiters = 10
                ok = converge!(op, ep, pars; fixρ=false,fixnorm=fixnorm)
                pars.ϵ = ϵ;
                pars.maxiters = maxiters
                ok = converge!(op, ep, pars; fixρ=false,fixnorm=fixnorm)
                tf = all_therm_func(op, ep)

                if ep.ρ < 0.3
                    ρ_right = -10
                    E_right = -1
                else
                    open(resfile, "a") do rf
                        println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op), " right")
                    end
                    ρ_right = ep.ρ
                    E_right = -tf.ϕ
                end
                done = true
            end

            push!(results, (ep.α, ep.Δ, ep.λ, ρ_left, ρ_right, E_left, E_right))

            open(resfile2, "a") do rf
                println(rf, ep.α, " ", ep.Δ, " ", ep.λ, " ", ρ_left, " ", ρ_right, " ", E_left, " ", E_right)
            end
            pars.verb > 0 && println(results[end])

    end
    return results
end

function phase_curves(results, outfile)
    res = readdlm(results)
    resmap = map(i->(res[i,:]...), 1:size(res,1))
    sort!(resmap)
    ### set condition !!!
    filter!(i->i[3]>=0.001,resmap)
    nosignal=filter(i->i[5]==-10, resmap)
    impossible=filter(i->i[4]<1 && i[5]>0. && i[6]>0 && i[7]>i[6], resmap)
    hard=filter(i->i[4] < 0.5 && i[5]>0.5 && i[7]>0 && i[7]<i[6], resmap)
    easy=filter(i->i[4]==10, resmap)
    f = open(outfile, "w")
    i = 0
    for l in (easy, hard, impossible)
        c = map(i->(i[3],i[1],i[2:end]...), l)
        sort!(c, rev=true)
        j = c[1][1]
        println(f, 3.1, " ", j*sqrt(1.15)," ", i)
        for k = 1:length(c)
            ck = c[k]
            ck[1] == j && continue
            j = ck[1]
            ckm1 = c[k-1]
            println(f, ckm1[2]-0.025/2," ", ckm1[1], " ", i)
        end
        println(f, c[end][2]-0.025/2," ", 0.00001, " ", i)
        i += 1
    end
    close(f)
end


function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    ep = ExtParams(res[line,1:4]...)
    tf = ThermFunc(res[line,5])
    op = OrderParams(res[line,6:end-1]...)
    return ep, op, tf
end

end ## module
