module PhaseRetr


using QuadGK
using AutoGrad
using Cubature
using Roots
using IterTools: product
# import LsqFit: curve_fit
using Distributed
using FastGaussQuadrature
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 10.0
const dx = .1

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

const interval2 = map(x->sign(x)*abs(x)^2, -1:0.01:1) .* 15
∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., atol=1e-7, rtol=1e-7, maxevals=10^3)[1]



∫Dlog(logf, int=interval2) = quadgk(z->begin # gaussian integration, but using logarithms to compare gaussian and integrated function
    lf, sf = logf(z)
    r = sf * exp(-z^2/2 + lf)/√(2π)
    isfinite(r) ? r : 0.0
end, int..., atol=1e7, rtol=1e7, maxevals=10^3)[1]


## Cubature.jl

∫∫D(f, xmin::Vector, xmax::Vector) = hcubature(z->begin
            ff = f(z[1],z[2])
            r = sign(ff) * exp(-(z[1]^2+z[2]^2)/2 + log(abs(ff)) - log(2π))
            isfinite(r) ? r : 0.0
        end, xmin, xmax, abstol=1e-9, reltol=1e-9, maxevals=10^3)[1]


∫∫D(fdim, f, xmin::Vector, xmax::Vector) = hcubature(fdim, (z,y)->begin
        r .= (G(z[1]).*G(z[2])).*f(z[1],z[2])
        isfinite(r) ? r : 0.0
    end, xmin, xmax, abstol=1e-7, maxevals=10^3)[1]

## Cuba.jl.
# ∫∫∫D(f, xmin::Vector, xmax::Vector) = cuhre((z,y)->begin
#             @. z = xmin + z*(xmax-xmin)
#             y[1] = G(z[1])*G(z[2])*G(z[3])*f(z[1],z[2],z[3])
#             # isfinite(r) ? r : 0.0
#         end, 3, 1,  abstol=1e-10)[1][1]*prod(xmax.-xmin)

function ∫∫D(f)
    ints = [(interval[i],interval[i+1]) for i=1:length(interval)-1]
    intprods = Iterators.product(ints, ints)
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

function ∫DD(f; n=51)
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
    m::Float64
    q0::Float64
    q1::Float64
    δq::Float64
    mh::Float64
    qh0::Float64
    qh1::Float64
    δqh::Float64
    s::Float64 # parisi breaking parameter
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
    Δ::Float64
    p::Float64
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

# Mathematica compatibility
Power(x,y) = x^y
Log(x) = log(x)


##################################
###### SADDLE-POINT APPROX #######
##################################

### example for F (perceptron LD)
# Fperc(z, cz1, h) = begin argz = cz1*z1 + h; dz=cz1; logH(argz) end
# dFperc(z, cz1, h) = begin argz = cz1*z + h; dz=cz1; -dz * GH(argz) end
# d2Fperc(z, cz1, h) = begin argz = cz1*z + h; dz=cz1; -dz^2 * GH(argz) * (GH(argz) - argz) end
# d3Fperc(z, cz1, h) = begin argz = cz1*z + h; dz=cz1; -dz^3 * (GH(argz) * (GH(argz) - argz)^2 + GH(argz) * (GH(argz) * (GH(argz) - argz) - 1)) end
# Fperc_derivatives(z, cz1, h) = begin argz = cz1*z + h; dz=cz1; gh = dz*GH(argz); (logH(argz), -gh, -gh * (gh - argz*dz), -gh * (gh - argz*dz)^2 - gh * (gh * (gh - argz*dz) - dz^2)) end
#
# function test1(large_prmt, cz1, h, dF, F_derivatives)
#     return (1/large_prmt)*log(∫Dlog(z1->(large_prmt*Fperc(z1, cz1, h),1)))
# end
# function test2(large_prmt, cz1, h, dF, F_derivatives, g)
#     return ∫Dlog(z1->begin
#         d = g(z1, cz1)
#         (large_prmt*Fperc(z1, cz1, h) + log(abs(d)), sign(d))
#     end) / ∫Dlog(z1->(large_prmt*Fperc(z1, cz1, h),1))
# end
function ord2_saddle_point_approx(large_prmt, cz1, h, (dF,d2F), F_derivatives; args=(h)) ### large_prmt >> 1 || large_prmt << -1
# function ord2_saddle_point_approx(large_prmt, cz1, h, dF, F_derivatives; args=(h)) ### large_prmt >> 1 || large_prmt << -1
    ### 2nd order saddle point approx for:
    #                       (1/large_prmt)*log(∫D(z1->F(cz1*z1 + h)^(large_prmt)))
    # (...assumes cz1 = O(1/√large_prmt))
    dff(z1) = - z1 + sign(large_prmt) * dF(z1, cz1*√(abs(large_prmt)), args...)
    # zs = find_zero(dff, 0.)
    ### second order zero search
    d2ff(z1) = -1 + sign(large_prmt) * d2F(z1, cz1*√(abs(large_prmt)), args...)
    # zs = find_zero((dff,d2ff),0.,Roots.Newton())
    zs = find_zero(dff,0.)
    ### expansion around maximum
    (Fs, dFs, d2Fs, d3Fs) = F_derivatives(zs, cz1*√(abs(large_prmt)), args...)
    f   = -zs^2/2 + sign(large_prmt) * Fs
    d1f = -zs + sign(large_prmt) * dFs
    d2f = -1 + sign(large_prmt) * d2Fs
    d3f = sign(large_prmt) * d3Fs
    return sign(large_prmt)*(f - 1/(2*abs(large_prmt))*log(-d2f)), zs, f, d1f, d2f, d3f
end
function ord2_saddle_point_approx_expectation(d2f, d3f, g, d1g, d2g, large_prmt) # g, d1g, d2g evaluated in zs*√(abs(large_prmt)) ### define g(cz1*z1) = g'(z1,cz1) (then evaluate derivatives in  (z1,cz1*√abs(large_prmt)))
    ### 2nd order saddle point approx for:
    #                           ∫D(z1-> F(cz1*z1 + h)^(large_prmt)* g(cz1*z1))
    #                          -------------------------------------------------------------------
    #                               ∫D(z1-> F(cz1*z1 + h)^(large_prmt))
    res = g + 1/(2*abs(large_prmt))*((d3f*d1g)/(d2f)^2 - (d2g)/d2f)
    return res
end

###################################################################################


#### INTERACTION TERM ####
function Gi(op, ep)
    @extract op: m q0 q1 δq qh0 qh1 δqh mh s
    (q1*δqh - 2*m*mh - δq*qh1 + q0*qh0*s - q1*qh1*s)/2
end

function ∂s_Gi(op, ep)
    @extract op: m q0 q1 δq qh0 qh1 δqh mh
    (q0*qh0 - q1*qh1) / 2
end

#### ENTROPIC TERM ####

function Gs(op)
    @extract op: qh0 qh1 δqh mh s
    0.5*((Power(mh,2) + qh0)/(δqh + (qh0 - qh1)*s) + Log(δqh)/s - Log(δqh + (qh0 - qh1)*s)/s)
end

function ∂s_Gs(op)
    @extract op: qh0 qh1 δqh mh s
    (-((qh0 - qh1)/(s*(δqh + s*(qh0 - qh1)))) -
    ((qh0 - qh1)*(qh0 + Power(mh,2)))/Power(δqh + s*(qh0 - qh1),2) -
    Log(δqh)/Power(s,2) + Log(δqh + s*(qh0 - qh1))/Power(s,2))/2
end

#### ENERGETIC TERM ####

### MSE for PhaseRetrieval like problems
fm(y, uu, p) = (y - abs(uu)^p)^2
d1fm(y, uu, p) = uu != 0 ? abs(uu)^p/(uu)*(2p*abs(uu)^p - p*2y) : 0.
d2fm(y, uu, p) = uu != 0 ? abs(uu)^(p)/(uu)^2*((2p-1)*2p*abs(uu)^p + p*(1-p)*2y) : Inf
d3fm(y, uu, p) = uu != 0 ? abs(uu)^(p)/(uu)^3*((2p-2)*(2p-1)*2p*abs(uu)^p - (2-p)*(1-p)*p*2y) : Inf
d1yfm(y, uu, p) = 2*(y - abs(uu)^p)
d2yfm(y, uu, p) = 2
d3yfm(y, uu, p) = 0
d1y_d1fm(y, uu, p) = uu != 0 ? -2p*abs(uu)^p/(uu) : 0.
d1y_d2fm(y, uu, p) = uu != 0 ?  p*(1-p)*2*abs(uu)^(p)/(uu)^2 : Inf

### for u maximization (β=∞)
fmax_u(u, h, y, cu, p) = -1/2*u^2 - fm(y, cu*u + h, p)
dfmax_u(u, h, y, cu, p) = -u - cu * d1fm(y, cu*u + h, p)
d2fmax_u(u, h, y, cu, p) = -1 - cu^2 * d2fm(y, cu*u + h, p)

### derivatives of max
d1fmax_h(h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(h, y, cu, p)[2]); us/cu end
d2fmax_h(h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(h, y, cu, p)[2]); d2=d2fm(y, cu*us + h, p); -d2/(1 + cu^2*d2) end
d3fmax_h(h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(h, y, cu, p)[2]); -d3fm(y, cu*us + h, p)/(1 + cu^2*d2fm(y, cu*us + h, p))^3 end

d1fmax_y(h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(h, y, cu, p)[2]); -2*(y - abs(cu*us+h)^p) end
# d2fmax_y(h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(h, y, cu, p)[2]); d2yfm(y, cu*us+h, p)-cu^2*d1y_d1fm(y, cu*us+h, p)^2/(1 + cu^2*d2fm(y, cu*us + h, p)) end
# d3fmax_y(h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(h, y, cu, p)[2]); d11=d1y_d1fm(y, cu*us + h, p); d2=d2fm(y, cu*us + h, p); -cu^4*d11^2/(1+cu^2*d2)^2*(-3*d1y_d2fm(y,cu*us+h,p)+cu^2*d11*d3fm(y,cu*us+h,p)/(1+cu^2*d2)) end

d1fmax_cu(h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(h, y, cu, p)[2]); (us^2/cu) end
# d2fmax_cu(h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(h, y, cu, p)[2]); d2=d2fm(y, cu*us + h, p); (d1fm(y, cu*us + h, p))^2/(1 + cu^2*d2)^2*(cu^2*d2*(2+3*cu^2*d2)-1) end
# d3fmax_cu(h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(h, y, cu, p)[2]); d1=d1fm(y, cu*us + h, p); d2=d2fm(y, cu*us + h, p); 12*cu*d1^2*d2/(1 + cu^2*d2)^2*(1 - cu^2*d2)-(2*cu*d1/(1 + cu^2*d2))^3*d3fm(y, cu*us + h, p)
#  end

### for saddle-point approximation (large s) in ∫Dz1, derivatives w.r.t. z1
d1fmax_z(z1, cz1, h, y, cu, p; us=-Inf) = cz1 * d1fmax_h(cz1*z1 + h, y, cu, p; us=us)
d2fmax_z(z1, cz1, h, y, cu, p; us=-Inf) = cz1^2 * d2fmax_h(cz1*z1 + h, y, cu, p; us=us)
d3fmax_z(z1, cz1, h, y, cu, p; us=-Inf) = cz1^3 * d3fmax_h(cz1*z1 + h, y, cu, p; us=us)
fmax_z_deriv(z1, cz1, h, y, cu, p) = begin fmax, us = max_argGe(cz1*z1+h, y, cu, p); (fmax, d1fmax_z(z1, cz1, h, y, cu, p; us=us), d2fmax_z(z1, cz1, h, y, cu, p; us=us), d3fmax_z(z1, cz1, h, y, cu, p; us=us)) end

fmax_dy(z1, cz1, h, y, cu, p; us=-Inf) = d1fmax_y(cz1*z1+h, y, cu, p; us=us)
d1fmax_dy(z1, cz1, h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(cz1*z1+h, y, cu, p)[2]); uu=cu*us+cz1*z1+h; 2p*cz1*abs(uu)^p/(uu)/(1+cu^2*d2fm(y,cu*us+cz1*z1+h,p))  end
d2fmax_dy(z1, cz1, h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(cz1*z1+h, y, cu, p)[2]); uu=cu*us+cz1*z1+h; d2=d2fm(y,uu,p); -2p*cz1^2*abs(uu)^p/(uu)*((1-p)/(uu)+cu^2/(1+cu^2*d2)*d3fm(y,uu,p))/(1+cu^2*d2)^2  end

fmax_dh(z1, cz1, h, y, cu, p; us=-Inf) = d1fmax_h(cz1*z1 + h, y, cu, p; us=us)
d1fmax_dh(z1, cz1, h, y, cu, p; us=-Inf) = cz1*d2fmax_h(cz1*z1 + h, y, cu, p; us=us)
d2fmax_dh(z1, cz1, h, y, cu, p; us=-Inf) = cz1^2*d3fmax_h(cz1*z1 + h, y, cu, p; us=us)

# fmax_dcz1(z1, cz1, mult, h, y, cu, p; us=-Inf) = mult*z1 * d1fmax_h(mult*cz1*z1 + h, y, cu, p; us=us)
# d1fmax_dcz1(z1, cz1, mult, h, y, cu, p; us=-Inf) = mult*(z1 * (mult*cz1)*d2fmax_h(mult*cz1*z1 + h, y, cu, p; us=us) + d1fmax_h(mult*cz1*z1 + h, y, cu, p; us=us))
# d2fmax_dcz1(z1, cz1, mult, h, y, cu, p; us=-Inf) = mult*(z1 * (mult*cz1)^2*d3fmax_h(mult*cz1*z1 + h, y, cu, p; us=us)+ (mult*cz1)*d2fmax_h(mult*cz1*z1 + h, y, cu, p; us=us) + (mult*cz1)*d2fmax_h(mult*cz1*z1 + h, y, cu, p; us=us))
fmax_dcz1(z1, cz1, mult, h, y, cu, p; us=-Inf) = mult*z1 * fmax_dh(z1, cz1*mult, h, y, cu, p; us=us)
d1fmax_dcz1(z1, cz1, mult, h, y, cu, p; us=-Inf) = mult*(z1 * d1fmax_dh(z1, cz1*mult, h, y, cu, p; us=us) + fmax_dh(z1, cz1*mult, h, y, cu, p; us=us))
d2fmax_dcz1(z1, cz1, mult, h, y, cu, p; us=-Inf) = mult*(z1 * d2fmax_dh(z1, cz1*mult, h, y, cu, p; us=us) + 2*d1fmax_dh(z1, cz1*mult, h, y, cu, p; us=us))

fmax_dcu(z1, cz1, h, y, cu, p; us=-Inf) = d1fmax_cu(cz1*z1+h, y, cu, p; us=us)
d1fmax_dcu(z1, cz1, h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(cz1*z1+h, y, cu, p)[2]); d2=d2fm(y, cu*us+cz1*z1+h, p); cz1*2us*(d2/(1 + cu^2*d2)) end
d2fmax_dcu(z1, cz1, h, y, cu, p; us=-Inf) = begin us==-Inf && (us = max_argGe(cz1*z1+h, y, cu, p)[2]); d2=d2fm(y, cu*us+cz1*z1+h, p);
    2cz1^2*(-cu*(d2/(1 + cu^2*d2))^2 + us*d3fm(y, cu*us+cz1*z1+h, p)/(1 + cu^2*d2)^3) end



function max_argGe(h, y, cu, p)
    ### find max of -1/2 u^2 - (y - abs(cu*u + h))^2
    @assert 1 <= p <= 2
    if abs(h) > - 2*cu^2*y
        us = 2*cu*(y-abs(h))*sign(h) / (1+2*cu^2)
        p==1 && (return -(y - abs(h))^2 / (1 + 2*cu^2), us)
        try
            us = find_zero((u->dfmax_u(u, h, y, cu, p), u->d2fmax_u(u, h, y, cu, p)), us, Roots.Newton())
            fmax_u(us, h, y, cu, p), us
        catch
            # us = 2*cu*(y-abs(h))*sign(h) / (1+2*cu^2)
            # fmax_u(us, h, y, cu, p), us
            # return -(y - abs(h))^2 / (1 + 2*cu^2), us
            NaN, NaN
        end
    else ### only for noisy case
        us = -h/cu ### p=1
        p==1 && (return -(h^2/(2*cu^2) + y^2), us)
        try
            us = find_zero(u->dfmax_u(u, h, y, cu, p), us)
            fmax_u(us, h, y, cu, p), us
        catch
            NaN, NaN
        end
    end
end

function argGe1RSB(y, h, cz1, δq, s, p)
    if p == 1
        c = cz1
        c2s = c^2*s
        d = 1+2δq
        b = (d+2c2s)
        g = (c*√(1+c2s/δq))
        g2 = (c*√(1+2c2s/d))
        return 1/s * log(if y < 0
                ### from cusp u*= -(cz1*z1 + h)/√δq
                exp(-s*h^2/(2δq*(1+c2s/δq))-s*y^2) * (H((h + 2*(δq + c2s)*y)/g) -
                H((h - 2*(δq + c2s)*y)/g)) / √(1+c2s/δq) +
                ### from minimum u*= 2√δq*(y-|cz1*z1 + h|)*sign(cz1*z1 + h) / (1+2δq)
                ( exp(-s*(h-y)^2/b) * H(-(h + 2(δq+c2s)*y) / g2) +
                exp(-s*(h+y)^2/b) * H( (h - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d)
            else
                ### from minimum u*= 2√δq*(y-|cz1*z1 + h|)*sign(cz1*z1 + h) / (1+2δq)
                ( exp(-s*(h-y)^2/b) * H(-(h*d + 2c2s*y) / (c*√(d*b))) +
                exp(-s*(h+y)^2/b) * H( (h*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d)
            end)#, 1/s * log(∫Dlog(z1->begin
            #     m, us = max_argGe(cz1*z1 + h, y, √δq, p)
            #     (s*m,1)
            # end)), ord2_saddle_point_approx(s, cz1, h, d1fmax_z, fmax_z_deriv; args=(h, y, √δq, p))[1]
    else
        if abs(s) > 10
            return ord2_saddle_point_approx(s, cz1, h, (d1fmax_z, d2fmax_z), fmax_z_deriv; args=(h, y, √δq, p))[1]
        else
            return 1/s * log(∫Dlog(z1->begin
                m, us = max_argGe(cz1*z1 + h, y, √δq, p)
                (s*m,1)
            end))
        end
    end
end

function Ge(op, ep)
    @extract op: m q0 q1 δq s
    @extract ep: Δ p
    # ∫D(u0->∫D(z0->begin
    #     Ge₀(abs(u0), m*u0 + √(q0 - m^2)*z0, q1-q0, δq, s)
    # end))
    if Δ == 0
        ge = ∫∫D((u0,z0)->begin
            argGe1RSB(abs(u0), m*u0 + √(q0 - m^2)*z0, √(q1-q0), δq, s, p)
        end)
    else
        a1 = √(Δ+(1-m^2/q0))
        b1 = (m/√q0)
        c1 = m/√q0*√(1/(1-m^2/q0)+ 1/Δ)
        d1 = (√(1-m^2/q0)/√Δ)
        # ge = 2∫D(z0->∫D(y->begin
        ge = 2*∫∫D((z0,y)->begin
            argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p) * H(-(c1*z0 + d1*y))
        end)
    end
    return ge
end


function Ge_test(op, ep)
    @extract op: m q0 q1 δq s
    @extract ep: Δ p
    # ∫D(u0->∫D(z0->begin
    #     Ge₀(abs(u0), m*u0 + √(q0 - m^2)*z0, q1-q0, δq, s)
    # end))
    # ge = ∫D(u0->∫D(z0->begin
    ge = ∫∫D((u0,z0)->begin
        ∂h_argGe1RSB(abs(u0), m*u0 + √(q0-m^2)*z0, √(q1-q0), δq, s, p) * 1/2*z0/√(q0-m^2)
    end)
        # a1 = √(Δ+(1-m^2/q0))
        # b1 = (m/√q0)
        # c1 = m/√q0*√(1/(1-m^2/q0)+ 1/Δ)
        # d1 = (√(1-m^2/q0)/√Δ)
        # # ge = 2∫D(z0->∫D(y->begin
        # ge = 2*∫∫D((z0,y)->begin
        #     argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p) * H(-(c1*z0 + d1*y))
        # end)
    return ge
end


function ∂y_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, s, p;  m=-Inf, zs=-Inf, fs=-Inf, d1fs=-Inf, d2fs=-Inf, d3fs=-Inf, mu=-Inf, us=-Inf)
    if p == 1
        a = sqrtq0_z0
        c = sqrt_q1_q0
        c2s = c^2*s
        d = 1+2δq
        b = (d+2c2s)
        g = (c*√(1+c2s/δq))
        g2 = (c*√(1+2c2s/d))
        return 1/s*(
            if y < 0
                exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (
                (-2s*y) * (H((a + 2*(δq + c2s)*y)/g) - H((a-2*(δq + c2s)*y)/g)) +
                (-G((a + 2*(δq + c2s)*y)/g) - G((a - 2*(δq + c2s)*y)/g)) * 2*(δq + c2s)/g
                ) / √(1+c2s/δq) +
              ( exp(-s*(a-y)^2/b) * (2s*(a-y)/b) *  H(-(a + 2(δq+c2s)*y) / g2) +
                exp(-s*(a+y)^2/b) * (-2s*(a+y)/b) * H( (a - 2(δq+c2s)*y) / g2) +
                (exp(-s*(a-y)^2/b) * G(-(a + 2(δq+c2s)*y) / g2) +
                 exp(-s*(a+y)^2/b) * G( (a - 2(δq+c2s)*y) / g2)) * 2(δq+c2s) / g2 ) / √(1+2c2s/d)
            else
              ( exp(-s*(a-y)^2/b) * (2s*(a-y)/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
                exp(-s*(a+y)^2/b) * (-2s*(a+y)/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d) +
                ( exp(-s*(a-y)^2/b) * G(-(a*d + 2c2s*y) / (c*√(d*b))) +
                  exp(-s*(a+y)^2/b) * G( (a*d - 2c2s*y) / (c*√(d*b))) ) * ( 2c2s / (c*√(d*b))) / √(1+2c2s/d)
            end
            ) /
            if y < 0
                exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (H((a + 2*(δq + c2s)*y)/g) -
                H((a - 2*(δq + c2s)*y)/g)) / √(1+c2s/δq) +
              ( exp(-s*(a-y)^2/b) * H(-(a + 2(δq+c2s)*y) / g2) +
                exp(-s*(a+y)^2/b) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d)

            else
              ( exp(-s*(a-y)^2/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
                exp(-s*(a+y)^2/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d)
            end
    else
        if abs(s) > 10
            if zs == -Inf
                m, zs, fs, d1fs, d2fs, d3fs = ord2_saddle_point_approx(s, sqrt_q1_q0, sqrtq0_z0, (d1fmax_z,d2fmax_z), fmax_z_deriv; args=(sqrtq0_z0, y, √δq, p))
                mu, us = max_argGe(sqrt_q1_q0*zs*√abs(s) + sqrtq0_z0, y, √δq, p)
            end
            g(z1, sqrt_q1_q0) = fmax_dy(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            d1g(z1, sqrt_q1_q0) = d1fmax_dy(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            d2g(z1, sqrt_q1_q0) = d2fmax_dy(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            return ord2_saddle_point_approx_expectation(d2fs, d3fs, g(zs, sqrt_q1_q0*√abs(s)), d1g(zs, sqrt_q1_q0*√abs(s)), d2g(zs, sqrt_q1_q0*√abs(s)), s)
        else
            return ∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    g = fmax_dy(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
                    (s*m + log(abs(g)), sign(g)) end) / ∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    (s*m, 1) end)
        end
    end
end

function ∂h_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, s, p;  m=-Inf, zs=-Inf, fs=-Inf, d1fs=-Inf, d2fs=-Inf, d3fs=-Inf, mu=-Inf, us=-Inf)
    if p==1
        a = sqrtq0_z0
        c = sqrt_q1_q0
        c2s = c^2*s
        d = 1+2δq
        b = (d+2c2s)
        g = (c*√(1+c2s/δq))
        g2 = (c*√(1+2c2s/d))

        1/s*(
        if y < 0
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (
            (-2s*a/(2δq*(1+c2s/δq))) * (H((a + 2*(δq + c2s)*y)/g) - H((a-2*(δq + c2s)*y)/g)) +
            (-G((a + 2*(δq + c2s)*y)/g) + G((a - 2*(δq + c2s)*y)/g)) / g
            ) / √(1+c2s/δq) +
          ( exp(-s*(a-y)^2/b) * (-2s*(a-y)/b) *  H(-(a + 2(δq+c2s)*y) / g2) +
            exp(-s*(a+y)^2/b) * (-2s*(a+y)/b) * H( (a - 2(δq+c2s)*y) / g2) +
            (exp(-s*(a-y)^2/b) * G(-(a + 2(δq+c2s)*y) / g2) -
             exp(-s*(a+y)^2/b) * G( (a - 2(δq+c2s)*y) / g2)) / g2 ) / √(1+2c2s/d)
        else
          ( exp(-s*(a-y)^2/b) * (-2s*(a-y)/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
            exp(-s*(a+y)^2/b) * (-2s*(a+y)/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d) +
            ( exp(-s*(a-y)^2/b) * G(-(a*d + 2c2s*y) / (c*√(d*b))) -
              exp(-s*(a+y)^2/b) * G( (a*d - 2c2s*y) / (c*√(d*b))) ) * (1 / (c*√(d*b))) / √(1+2c2s/d)
        end
        ) /
        if y < 0
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (H((a + 2*(δq + c2s)*y)/g) -
            H((a - 2*(δq + c2s)*y)/g)) / √(1+c2s/δq) +
          ( exp(-s*(a-y)^2/b) * H(-(a + 2(δq+c2s)*y) / g2) +
            exp(-s*(a+y)^2/b) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d)

        else
          ( exp(-s*(a-y)^2/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
            exp(-s*(a+y)^2/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d)
        end
    else
        if abs(s) > 10
            if zs == -Inf
                m, zs, fs, d1fs, d2fs, d3fs = ord2_saddle_point_approx(s, sqrt_q1_q0, sqrtq0_z0, (d1fmax_z, d2fmax_z), fmax_z_deriv; args=(sqrtq0_z0, y, √δq, p))
                mu, us = max_argGe(sqrt_q1_q0*zs*√abs(s) + sqrtq0_z0, y, √δq, p)
            end
            g(z1, sqrt_q1_q0) = fmax_dh(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            d1g(z1, sqrt_q1_q0) = d1fmax_dh(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            d2g(z1, sqrt_q1_q0) = d2fmax_dh(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            return ord2_saddle_point_approx_expectation(d2fs, d3fs, g(zs, sqrt_q1_q0*√abs(s)), d1g(zs, sqrt_q1_q0*√abs(s)), d2g(zs, sqrt_q1_q0*√abs(s)), s)
        else
            return ∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    g = fmax_dh(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
                    (s*m + log(abs(g)), sign(g)) end) / ∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    (s*m, 1) end)
        end
    end
end

function ∂cz1_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, s, p;  m=-Inf, zs=-Inf, fs=-Inf, d1fs=-Inf, d2fs=-Inf, d3fs=-Inf, mu=-Inf, us=-Inf)
    if p == 1
        a = sqrtq0_z0
        c = sqrt_q1_q0
        c2s = c^2*s
        d = 1+2δq
        b = (d+2c2s)
        g = (c*√(1+c2s/δq))
        g2 = (c*√(1+2c2s/d))

        1/s*(
        if y < 0
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) *
            ((4*s^2*a^2/(2δq*(1+c2s/δq))^2*c) * (H((a + 2*(δq + c2s)*y)/g) - H((a-2*(δq + c2s)*y)/g)) +
            - G((a + 2*(δq + c2s)*y)/g) * (4*y*c*s/g - (a + 2*(δq + c2s)*y)/g^2 * (√(1+c2s/δq) + c/√(1+c2s/δq)*s*c/δq)) +
            + G((a - 2*(δq + c2s)*y)/g) * (-4*y*c*s/g - (a - 2*(δq + c2s)*y)/g^2 * (√(1+c2s/δq) + c/√(1+c2s/δq)*s*c/δq))
            ) / √(1+c2s/δq) +
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (H((a + 2*(δq + c2s)*y)/g) -
            H((a - 2*(δq + c2s)*y)/g)) / √(1+c2s/δq) * (-1/(1+c2s/δq)*s*c/δq) +
            ( exp(-s*(a-y)^2/b) * (4*c*s^2*(a-y)^2/b^2) * H(-(a + 2(δq+c2s)*y) / g2) +
            exp(-s*(a+y)^2/b) * (4*c*s^2*(a+y)^2/b^2) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d) -
            (exp(-s*(a-y)^2/b) * G(-(a + 2(δq+c2s)*y) / g2) * (-4c*s*y/g2 + (a+2(δq+c2s)*y)/g2^2 * (√(1+2c2s/d) + c/√(1+2c2s/d)*2*c*s/d))  +
             exp(-s*(a+y)^2/b) * G( (a - 2(δq+c2s)*y) / g2) * (-4c*s*y/g2 - (a-2(δq+c2s)*y)/g2^2 * (√(1+2c2s/d) + c/√(1+2c2s/d)*2*c*s/d))
             ) / √(1+2c2s/d) +
             ( exp(-s*(a-y)^2/b) * H(-(a + 2(δq+c2s)*y) / g2) +
               exp(-s*(a+y)^2/b) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d) * (-1/(1+2c2s/d)*2*c*s/d)
        else
            ( exp(-s*(a-y)^2/b) * (4c*s^2*(a-y)^2/b^2) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
              exp(-s*(a+y)^2/b) * (4c*s^2*(a+y)^2/b^2) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d) +
            ( -exp(-s*(a-y)^2/b) * G(-(a*d + 2c2s*y) / (c*√(d*b))) * (-(4c*s*y) / (c*√(d*b)) + (a*d + 2c2s*y) / (c*√(d*b))^2 * (√(d*b) + 2*c/√(d*b)*d*c*s)) -
              exp(-s*(a+y)^2/b) * G( (a*d - 2c2s*y) / (c*√(d*b))) * (-(4c*s*y) / (c*√(d*b)) - (a*d - 2c2s*y) / (c*√(d*b))^2 * (√(d*b) + 2*c/√(d*b)*d*c*s))) / √(1+2c2s/d) +
              ( exp(-s*(a-y)^2/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
                exp(-s*(a+y)^2/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d) * (-1/(1+2c2s/d)*2*c*s/d)
        end
        ) /
        if y < 0
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (H((a + 2*(δq + c2s)*y)/g) -
            H((a - 2*(δq + c2s)*y)/g)) / √(1+c2s/δq) +
          ( exp(-s*(a-y)^2/b) * H(-(a + 2(δq+c2s)*y) / g2) +
            exp(-s*(a+y)^2/b) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d)

        else
          ( exp(-s*(a-y)^2/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
            exp(-s*(a+y)^2/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d)
        end
    else
        if abs(s) > 10
            if zs == -Inf
                m, zs, fs, d1fs, d2fs, d3fs = ord2_saddle_point_approx(s, sqrt_q1_q0, sqrtq0_z0, (d1fmax_z, d2fmax_z), fmax_z_deriv; args=(sqrtq0_z0, y, √δq, p))
                mu, us = max_argGe(sqrt_q1_q0*zs*√abs(s) + sqrtq0_z0, y, √δq, p)
            end
            g(z1, mult) = fmax_dcz1(z1, sqrt_q1_q0, mult, sqrtq0_z0, y, √δq, p; us=us)
            d1g(z1, mult) = d1fmax_dcz1(z1, sqrt_q1_q0, mult, sqrtq0_z0, y, √δq, p; us=us)
            d2g(z1, mult) = d2fmax_dcz1(z1, sqrt_q1_q0, mult, sqrtq0_z0, y, √δq, p; us=us)
            return ord2_saddle_point_approx_expectation(d2fs, d3fs, g(zs,√abs(s)), d1g(zs,√abs(s)), d2g(zs,√abs(s)), s)
        else
            return ∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    g = fmax_dcz1(z1, sqrt_q1_q0, 1, sqrtq0_z0, y, √δq, p; us=us)
                    (s*m + log(abs(g)), sign(g)) end) / ∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    (s*m, 1) end)
        end
    end
end

function ∂δq_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, s, p;  m=-Inf, zs=-Inf, fs=-Inf, d1fs=-Inf, d2fs=-Inf, d3fs=-Inf, mu=-Inf, us=-Inf)
    if p == 1
        a = sqrtq0_z0
        c = sqrt_q1_q0
        c2s = c^2*s
        d = 1+2δq
        b = (d+2c2s)
        g = (c*√(1+c2s/δq))
        g2 = (c*√(1+2c2s/d))

        1/s*(
        if y < 0
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (
            (2*s*a^2/(2δq*(1+c2s/δq))^2) * (H((a + 2*(δq + c2s)*y)/g) - H((a-2*(δq + c2s)*y)/g)) +
            - G((a + 2*(δq + c2s)*y)/g) * (2*y/g - (a + 2*(δq + c2s)*y)/g^2 * (1/2*c/√(1+c2s/δq)*(-c2s/δq^2))) +
            + G((a - 2*(δq + c2s)*y)/g) * (-2*y/g - (a - 2*(δq + c2s)*y)/g^2 * (1/2*c/√(1+c2s/δq)*(-c2s/δq^2)))
            ) / √(1+c2s/δq) +
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (H((a + 2*(δq + c2s)*y)/g) -
            H((a - 2*(δq + c2s)*y)/g)) / √(1+c2s/δq) * (-1/2/(1+c2s/δq)*(-c2s/δq^2)) +
            ( exp(-s*(a-y)^2/b) * (2*s*(a-y)^2/b^2) * H(-(a + 2(δq+c2s)*y) / g2) +
              exp(-s*(a+y)^2/b) * (2*s*(a+y)^2/b^2) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d) -
            (exp(-s*(a-y)^2/b) * G(-(a + 2(δq+c2s)*y) / g2) * (-2*y/g2 + (a+2(δq+c2s)*y)/g2^2 * (-c/√(1+2c2s/d)*2*c2s/d^2))  +
             exp(-s*(a+y)^2/b) * G( (a - 2(δq+c2s)*y) / g2) * (-2*y/g2 - (a-2(δq+c2s)*y)/g2^2 * (-c/√(1+2c2s/d)*2*c2s/d^2))
             ) / √(1+2c2s/d) +
             ( exp(-s*(a-y)^2/b) * H(-(a + 2(δq+c2s)*y) / g2) +
               exp(-s*(a+y)^2/b) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d) * (1/(1+2c2s/d)*2*c2s/d^2)
        else
            ( exp(-s*(a-y)^2/b) * (2s*(a-y)^2/b^2) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
              exp(-s*(a+y)^2/b) * (2s*(a+y)^2/b^2) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d) +
            ( -exp(-s*(a-y)^2/b) * G(-(a*d + 2c2s*y) / (c*√(d*b))) * (-(2a) / (c*√(d*b)) + (a*d + 2c2s*y) / (c*√(d*b))^2 * (1/2*c/√(d*b)*(2d+2b))) -
              exp(-s*(a+y)^2/b) * G( (a*d - 2c2s*y) / (c*√(d*b))) * ((2a) / (c*√(d*b)) - (a*d - 2c2s*y) / (c*√(d*b))^2 * (1/2*c/√(d*b)*(2d+2b)))) / √(1+2c2s/d) +
              ( exp(-s*(a-y)^2/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
                exp(-s*(a+y)^2/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d) * (1/(1+2c2s/d)*2c2s/d^2)
        end
        ) /
        if y < 0
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (H((a + 2*(δq + c2s)*y)/g) -
            H((a - 2*(δq + c2s)*y)/g)) / √(1+c2s/δq) +
          ( exp(-s*(a-y)^2/b) * H(-(a + 2(δq+c2s)*y) / g2) +
            exp(-s*(a+y)^2/b) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d)
        else
          ( exp(-s*(a-y)^2/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
            exp(-s*(a+y)^2/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d)
        end
    else
        if abs(s) > 10
            if zs == -Inf
                m, zs, fs, d1fs, d2fs, d3fs = ord2_saddle_point_approx(s, sqrt_q1_q0, sqrtq0_z0, (d1fmax_z, d2fmax_z), fmax_z_deriv; args=(sqrtq0_z0, y, √δq, p))
                mu, us = max_argGe(sqrt_q1_q0*zs*√abs(s) + sqrtq0_z0, y, √δq, p)
            end
            g(z1, sqrt_q1_q0) = fmax_dcu(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            d1g(z1, sqrt_q1_q0) = d1fmax_dcu(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            d2g(z1, sqrt_q1_q0) = d2fmax_dcu(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            return 1/(2*√δq)*ord2_saddle_point_approx_expectation(d2fs, d3fs, g(zs, sqrt_q1_q0*√abs(s)), d1g(zs, sqrt_q1_q0*√abs(s)), d2g(zs, sqrt_q1_q0*√abs(s)), s)
        else
            return ∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    g = 1/(2*√δq)*fmax_dcu(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
                    (s*m + log(abs(g)), sign(g)) end) / ∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    (s*m, 1) end)
        end
    end
end

function ∂s_argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, s, p;  m=-Inf, zs=-Inf, fs=-Inf, d1fs=-Inf, d2fs=-Inf, d3fs=-Inf, mu=-Inf, us=-Inf)
    if p==1
        a = sqrtq0_z0
        c = sqrt_q1_q0
        c2s = c^2*s
        d = 1+2δq
        b = (d+2c2s)
        g = (c*√(1+c2s/δq))
        g2 = (c*√(1+2c2s/d))

        1/s*(
        if y < 0
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (
            (-a^2/(2δq*(1+c2s/δq)) + s*a^2/(2δq*(1+c2s/δq))^2*2c^2-y^2) * (H((a + 2*(δq + c2s)*y)/g) - H((a-2*(δq + c2s)*y)/g)) +
            - G((a + 2*(δq + c2s)*y)/g) * (2*y*c^2/g - (a + 2*(δq + c2s)*y)/g^2 * (1/2*c^3/√(1+c2s/δq)/δq)) +
            + G((a - 2*(δq + c2s)*y)/g) * (-2*y*c^2/g - (a - 2*(δq + c2s)*y)/g^2 * (1/2*c^3/√(1+c2s/δq)/δq))
            ) / √(1+c2s/δq) +
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (H((a + 2*(δq + c2s)*y)/g) -
            H((a - 2*(δq + c2s)*y)/g)) / √(1+c2s/δq) * (-1/2/(1+c2s/δq)*c^2/δq) +

            ( exp(-s*(a-y)^2/b) * (-(a-y)^2/b + s*2*c^2*(a-y)^2/b^2) * H(-(a + 2(δq+c2s)*y) / g2) +
            exp(-s*(a+y)^2/b) * (-(a+y)^2/b + s*2*c^2*(a+y)^2/b^2) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d) -
            (exp(-s*(a-y)^2/b) * G(-(a + 2(δq+c2s)*y) / g2) * (-2c^2*y/g2 + (a+2(δq+c2s)*y)/g2^2 * (c^3/√(1+2c2s/d)/d))  +
             exp(-s*(a+y)^2/b) * G( (a - 2(δq+c2s)*y) / g2) * (-2c^2*y/g2 - (a-2(δq+c2s)*y)/g2^2 * (c^3/√(1+2c2s/d)/d))
             ) / √(1+2c2s/d) +
             ( exp(-s*(a-y)^2/b) * H(-(a + 2(δq+c2s)*y) / g2) +
               exp(-s*(a+y)^2/b) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d) * (-1/(1+2c2s/d)*c^2/d)
        else
            ( exp(-s*(a-y)^2/b) * (-(a-y)^2/b + s*2*c^2*(a-y)^2/b^2) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
              exp(-s*(a+y)^2/b) * (-(a+y)^2/b + s*2*c^2*(a+y)^2/b^2) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d) +
            ( -exp(-s*(a-y)^2/b) * G(-(a*d + 2c2s*y) / (c*√(d*b))) * (-(2c^2*y) / (c*√(d*b)) + (a*d + 2c2s*y) / (c*√(d*b))^2 * (c/√(d*b)*d*c^2)) -
              exp(-s*(a+y)^2/b) * G( (a*d - 2c2s*y) / (c*√(d*b))) * (-(2c^2*y) / (c*√(d*b)) - (a*d - 2c2s*y) / (c*√(d*b))^2 * (c/√(d*b)*d*c^2))) / √(1+2c2s/d) +
              ( exp(-s*(a-y)^2/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
                exp(-s*(a+y)^2/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d) * (-1/(1+2c2s/d)*c^2/d)
        end
        ) /
        if y < 0
            exp(-s*a^2/(2δq*(1+c2s/δq))-s*y^2) * (H((a + 2*(δq + c2s)*y)/g) -
            H((a - 2*(δq + c2s)*y)/g)) / √(1+c2s/δq) +
          ( exp(-s*(a-y)^2/b) * H(-(a + 2(δq+c2s)*y) / g2) +
            exp(-s*(a+y)^2/b) * H( (a - 2(δq+c2s)*y) / g2) ) / √(1+2c2s/d)

        else
          ( exp(-s*(a-y)^2/b) * H(-(a*d + 2c2s*y) / (c*√(d*b))) +
            exp(-s*(a+y)^2/b) * H( (a*d - 2c2s*y) / (c*√(d*b))) ) / √(1+2c2s/d)
        end +
        -1/s * argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, s, p)
    else
        if abs(s) > 10
            if zs == -Inf
                m, zs, fs, d1fs, d2fs, d3fs = ord2_saddle_point_approx(s, sqrt_q1_q0, sqrtq0_z0, (d1fmax_z, d2fmax_z), fmax_z_deriv; args=(sqrtq0_z0, y, √δq, p))
                mu, us = max_argGe(sqrt_q1_q0*zs*√abs(s) + sqrtq0_z0, y, √δq, p)
            end
            g(z1, sqrt_q1_q0) = mu
            d1g(z1, sqrt_q1_q0) = d1fmax_z(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            d2g(z1, sqrt_q1_q0) = d2fmax_z(z1, sqrt_q1_q0, sqrtq0_z0, y, √δq, p; us=us)
            return 1/s*(ord2_saddle_point_approx_expectation(d2fs, d3fs, g(zs, sqrt_q1_q0*√abs(s)), d1g(zs, sqrt_q1_q0*√abs(s)), d2g(zs, sqrt_q1_q0*√abs(s)), s) - m)
        else
            return 1/s*(∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    g = m
                    (s*m + log(abs(g)), sign(g)) end) / ∫Dlog(z1->begin
                    m, us = max_argGe(sqrt_q1_q0*z1 + sqrtq0_z0, y, √δq, p)
                    (s*m, 1) end) - argGe1RSB(y, sqrtq0_z0, sqrt_q1_q0, δq, s, p))
        end
    end
end

#### CHECKED!



function ∂q0_Ge(op, ep)
    @extract op: q0 q1 δq m s
    @extract ep: Δ p
    if Δ == 0
        ms, zs, fs, d1fs, d2fs, d3fs, mu, us = (-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf)
        ge = ∫∫D((u0,z0)->begin
            if abs(s) > 10 && p>1
                ms, zs, fs, d1fs, d2fs, d3fs = ord2_saddle_point_approx(s, √(q1-q0), m*u0+√(q0-m^2)*z0, (d1fmax_z,d2fmax_z), fmax_z_deriv; args=(m*u0 + √(q0-m^2)*z0, abs(u0), √δq, p))
                mu, us = max_argGe(√(q1-q0)*zs*√abs(s) + m*u0 + √(q0-m^2)*z0, abs(u0), √δq, p)
            end
            ∂h_argGe1RSB(abs(u0), m*u0+√(q0-m^2)*z0, √(q1-q0), δq, s, p; m=ms, zs=zs, fs=fs, d1fs=d1fs, d2fs=d2fs, d3fs=d3fs, mu=mu, us=us)*(1/(2*√(q0-m^2)))*z0+
            ∂cz1_argGe1RSB(abs(u0), m*u0 + √(q0-m^2)*z0, √(q1-q0), δq, s, p; m=ms, zs=zs, fs=fs, d1fs=d1fs, d2fs=d2fs, d3fs=d3fs, mu=mu, us=us)*(-1/(2*√(q1-q0)))
        end)
        # δ = min(1e-6, (op.q1-op.q0)/2, (op.q0-op.m^2)/2)
        # op.q0 += δ
        # ge1 = Ge(op,ep)
        # op.q0 -= 2δ
        # ge2 = Ge(op,ep)
        # op.q0 += δ
        # return (ge1 - ge2) / 2δ
    else
        # δ = min(1e-5, (op.q1-op.q0)/2, (op.q0-op.m^2)/2)
        # op.q0 += δ
        # ge1 = Ge(op,ep)
        # op.q0 -= 2δ
        # ge2 = Ge(op,ep)
        # op.q0 += δ
        # return (ge1 - ge2) / 2δ
        a1 = √(Δ+(1-m^2/q0))
        b1 = (m/√q0)
        da1 = 1/(2*a1)*m^2/q0^2
        db1 = -1/2*b1/q0
        c1 = m/√q0*√(1/(1-m^2/q0)+ 1/Δ)
        dc1 = -1/2*c1/q0 -1/2*m/√q0/√(1/(1-m^2/q0)+ 1/Δ)*1/(1-m^2/q0)^2*m^2/q0^2
        d1 = (√(1-m^2/q0)/√Δ)
        dd1 = 1/2/√(1-m^2/q0)/√Δ*m^2/q0^2
        # ge = 2∫DD(z0->∫DD(y->begin
        ms, zs, fs, d1fs, d2fs, d3fs, mu, us = (-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf,-Inf)
        ge = 2∫∫D((z0,y)->begin
            if abs(s) > 10 && p>1
                ms, zs, fs, d1fs, d2fs, d3fs = ord2_saddle_point_approx(s, √(q1-q0), √q0*z0, (d1fmax_z,d2fmax_z), fmax_z_deriv; args=(√q0*z0, a1*y + b1*z0, √δq, p))
                mu, us = max_argGe( √(q1-q0)*zs*√abs(s) + √q0*z0, a1*y + b1*z0, √δq, p)
            end
            ∂y_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p; m=ms, zs=zs, fs=fs, d1fs=d1fs, d2fs=d2fs, d3fs=d3fs, mu=mu, us=us) * (da1*y + db1*z0) * H(-(c1*z0 + d1*y)) +
            ∂h_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p; m=ms, zs=zs, fs=fs, d1fs=d1fs, d2fs=d2fs, d3fs=d3fs, mu=mu, us=us) * (1/2*z0/√q0) * H(-(c1*z0 + d1*y)) +
            ∂cz1_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p; m=ms, zs=zs, fs=fs, d1fs=d1fs, d2fs=d2fs, d3fs=d3fs, mu=mu, us=us) * (-1/(2*√(q1-q0))) * H(-(c1*z0 + d1*y)) +
            argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p) * G(-(c1*z0 + d1*y)) * (dc1*z0 + dd1*y)
        end)
    end
end

function ∂q1_Ge(op, ep)
    @extract op: m q0 q1 δq s
    @extract ep: Δ p

    if Δ == 0
        return ∫∫D((u0,z0)->begin
            ∂cz1_argGe1RSB(abs(u0), m*u0 + √(q0 - m^2)*z0, √(q1-q0), δq, s, p)* (1/(2*√(q1-q0)))
        end)
        # q1s = op.q1
        # δ = min(1e-6, (op.q1-op.q0)/2)
        # op.q1 = q1s+δ
        # ge1 = Ge(op,ep)
        # op.q1 = q1s-δ
        # ge2 = Ge(op,ep)
        # op.q1 = q1s
        # return (ge1 - ge2) / 2δ
    else
        a1 = √(Δ+(1-m^2/q0))
        b1 = (m/√q0)
        c1 = m/√q0*√(1/(1-m^2/q0)+ 1/Δ)
        d1 = (√(1-m^2/q0)/√Δ)
        ge = 2∫∫D((z0,y)->begin
            ∂cz1_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p) * (1/(2*√(q1-q0))) * H(-(c1*z0 + d1*y))
        end)
        return ge
    end
end

function ∂δq_Ge(op, ep)
    @extract op: m q0 q1 δq s
    @extract ep: Δ p
    if Δ == 0
        return ∫∫D((u0,z0)->begin
            ∂δq_argGe1RSB(abs(u0), m*u0 + √(q0 - m^2)*z0, √(q1-q0), δq, s, p)
        end)
    else
        a1 = √(Δ+(1-m^2/q0))
        b1 = (m/√q0)
        c1 = m/√q0*√(1/(1-m^2/q0)+ 1/Δ)
        d1 = (√(1-m^2/q0)/√Δ)
        ge = 2∫∫D((z0,y)->begin
            ∂δq_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p) * H(-(c1*z0 + d1*y))
        end)
    end
end

function ∂m_Ge(op, ep)
    @extract op: m q0 q1 δq s
    @extract ep: Δ p
    if Δ == 0
        return ∫∫D((u0,z0)->begin
            ∂h_argGe1RSB(abs(u0), m*u0 + √(q0-m^2)*z0, √(q1-q0), δq, s, p) * (u0 - m/√(q0 - m^2)*z0)
            # ∂y_argGe1RSB(abs(√(1-m^2/q0)*u0 + m/√q0*z0), √q0*z0, √(q1-q0), δq, s, p) * sign(√(1-m^2/q0)*u0 + m/√q0*z0)*(-m/q0*u0/√(1-m^2/q0) + 1/√q0*z0)
        end)
        # mm = op.m
        # op.m = mm + 1e-6
        # ge1 = Ge(op,ep)
        # op.m = mm - 1e-6
        # ge2 = Ge(op,ep)
        # op.m = mm
        # return (ge1 - ge2) / 2e-6
    else
        a1 = √(Δ+(1-m^2/q0))
        da1 = -1/(a1)*m/q0
        b1 = (m/√q0)
        db1 = 1/√q0
        c1 = m/√q0*√(1/(1-m^2/q0)+ 1/Δ)
        dc1 = c1/m + m/√q0/√(1/(1-m^2/q0)+ 1/Δ)*1/(1-m^2/q0)^2*m/q0
        d1 = (√(1-m^2/q0)/√Δ)
        dd1 = -1/√(1-m^2/q0)/√Δ*m/q0
        ge = 2∫∫D((z0,y)->begin
            ∂y_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p) * (da1*y + db1*z0) * H(-(c1*z0 + d1*y)) +
            argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p) * G(-(c1*z0 + d1*y)) * (dc1*z0 + dd1*y)
        end)
    end
end

function ∂s_Ge(op, ep)
    @extract op: m q0 q1 δq s
    @extract ep: Δ p
    if Δ == 0
        # return 2∫DD(z0->∫DD(u0->begin
        return ∫∫D((u0,z0)->begin
            ∂s_argGe1RSB(abs(u0), m*u0 + √(q0 - m^2)*z0, √(q1-q0), δq, s, p)
        end)
        # op.s += 1e-6
        # ge1 = Ge(op,ep)
        # op.s -= 2e-6
        # ge2 = Ge(op,ep)
        # op.s += 1e-6
        # return (ge1 - ge2) / 2e-6
    else
        a1 = √(Δ+(1-m^2/q0))
        b1 = (m/√q0)
        c1 = m/√q0*√(1/(1-m^2/q0)+ 1/Δ)
        d1 = (√(1-m^2/q0)/√Δ)
        # ge = 2∫DD(z0->∫DD(y->begin
        ge = 2∫∫D((z0,y)->begin
            ∂s_argGe1RSB(a1*y + b1*z0, √q0*z0, √(q1-q0), δq, s, p) * H(-(c1*z0 + d1*y))
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
fqh0(op, ep) = -2/op.s * ep.α * ∂q0_Ge(op, ep)
fqh1(op, ep) = 2ep.α * ∂δq_Ge(op, ep)
fδqh(op, ep) = op.qh1*op.s -2ep.α * ∂q1_Ge(op, ep)
fρh(op, ep) = ep.α * ∂m_Ge(op, ep)

function fq0(op)
    @extract op: qh0 qh1 δqh mh s
    (qh0 + mh^2) / (δqh + s*(qh0 - qh1))^2
end

function fδq(op)
    @extract op: qh0 qh1 δqh mh s q1
    - s*q1 + 1/(δqh + s*(qh0 - qh1)) + (s*(qh0 + Power(mh,2)))/Power(δqh + s*(qh0 - qh1),2)
end
function fq1(op)
    @extract op: qh0 qh1 δqh mh s q1
    -(1/(δqh*s) - (qh0 + Power(mh,2))/Power(δqh + (qh0 - qh1)*s,2) -
      1/(s*(δqh + (qh0 - qh1)*s)))
end
function fm(op)
    @extract op: qh0 qh1 δqh mh s q1
    mh/(δqh + s*(qh0 - qh1))
end

function imh(op, ep)
    @extract op: m qh0 qh1 δqh s q1
    (true, m*(δqh + s*(qh0 - qh1)))
end
function iδqh_fun(δq, op)
    @extract op: qh0 qh1 δqh mh s q1
    0.5*q1 + (1/(δqh*s) - 1/(s*(δqh + s*(qh0 - qh1))) -
    (qh0 + Power(mh,2))/Power(δqh + s*(qh0 - qh1),2))/2.
end

function iδqh(op, δqh₀, atol=1e-10)
    ok, δqh, it, normf0 = findroot(δqh -> iδqh_fun(δqh, op), δqh₀, NewtonMethod(atol=atol))
    ok || error("iδqh failed: iδqh=$(δqh), it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, δqh
end


function is_fun(op::OrderParams, ep::ExtParams, s)
    @extract op: q0 q1 δq qh0 qh1 δqh mh
    @extract ep: α
    op.s = s # for Ge
    ∂s_Gi(op, ep) + ∂s_Gs(op) + α*∂s_Ge(op, ep)
end

function is(op::OrderParams, ep::ExtParams, s₀, atol=1e-4)
    ok, s, it, normf0 = findroot(s -> is_fun(op, ep, s), s₀, NewtonMethod(atol=atol))
    ok || error("im failed: m=$m, it=$it, normf0=$normf0")
    return ok, s
end


###############################
function fhats_slow(op, ep)
    qh0 = qh1 = δqh = 0
    @sync begin
        qh0 = @spawn fqh0(op, ep)
        qh1 = @spawn fqh1(op, ep)
        δqh = @spawn fδqh(op, ep)
        mh = @spawn fmh(op, ep)
    end
    return fetch(qh0), fetch(qh1), (fetch(qh1)-op.qh1)*op.s + fetch(δqh), fetch(mh)
end

function fix_inequalities!(op, ep)
    if op.q0 < op.m^2
        op.q0 = op.m^2 + rand()*1e-4
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
    if op.δqh + (op.qh0 - op.qh1)*op.s < 0
        op.δqh = (op.qh0 - op.qh1)*op.s + rand()*1e-4
    end
end

function converge!(op::OrderParams, ep::ExtParams, pars::Params;
                   testm = false,
        fixm=true, fixnorm=true, fixs=true, extrap=-1)
    @extract pars: maxiters verb ϵ ψ

    Δ = Inf
    ok = false
    fix_inequalities!(op, ep)

    ops = Vector{OrderParams}() # keep some history and extrapolate for quicker convergence
    for it = 1:maxiters
        Δ = 0.0
        ok = oki = true
        verb > 1 && println("it=$it")


        qh0, qh1, δqh, mh = fhats_slow(op, ep)

        @update  op.qh0    identity       Δ ψ verb  qh0
        @update  op.qh1    identity       Δ ψ verb  qh1
        if fixnorm
            @updateI op.δqh ok   iδqh     Δ ψ verb  op ep
        else
            @update  op.δqh    identity     Δ ψ verb  δqh
        end
        if fixm
            @updateI op.mh oki   imh   Δ ψ verb  op ep
            ok &= oki
        else
            @update  op.mh  identity    Δ ψ verb  mh
        end

        # fix_inequalities_hat!(op, ep)
        fix_inequalities!(op, ep)

        @update op.q0   fq0       Δ ψ verb  op
        if !fixnorm
            @update op.q1   fq1     Δ ψ verb  op
        end
        @update op.δq   fδq       Δ ψ verb  op
        if !fixm
            @update op.m   fm     Δ ψ verb  op
        end

        if !fixs
            @updateI op.s oki   is    Δ ψ verb  op ep op.s
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
        fixρ=fixρ,fixnorm=fixnorm,extrap=extrap,fixm=fixm)
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
