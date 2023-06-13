module Perc

using LittleScienceTools.Roots
using QuadGK
using LsqFit
using AutoGrad
# using FastGaussQuadrature
include("../../common.jl")


###### INTEGRATION  ######
const ∞ = 30.0
const dx = 0.002

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

## quadgk
∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-8, maxevals=10^5)[1]


∫Dexp(f, g=z->1, int=interval) = quadgk(z->begin
    r = logG(z) + f(z)
    r = exp(r) * g(z)
end, int..., abstol=1e-8, maxevals=10^7)[1]

## hermite polynomials
# let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
#     global gw
#     gw(n::Int) = get!(s, n) do
#         (x,w) = gausshermite(n)
#         return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
#     end
# end
# function ∫D(f; n=1000)
#     (xs, ws) = gw(n)
#     s = 0.0
#     for (x,w) in zip(xs, ws)
#         y = f(x)
#         s += w  * ifelse(isfinite(y), y, 0.0)
#     end
#     return s
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



############### PARAMS ################

mutable struct OrderParams
    q0::Float64
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
Sqrt(x) = sqrt(x)

#### INTERACTION TERM ####
Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) = (δqh - 2*ρ*ρh - δq*qh1 + q0*qh0*m - qh1*m)/2

∂m_Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) = (q0*qh0 - qh1) / 2

#### ENTROPIC TERM ####

Gs(qh0,qh1,δqh,ρh,m) = 0.5*((Power(ρh,2) + qh0)/(δqh + (qh0 - qh1)*m) + Log(δqh)/m - Log(δqh + (qh0 - qh1)*m)/m)

∂m_Gs(qh0,qh1,δqh,ρh,m) =  (-((qh0 - qh1)/(m*(δqh + m*(qh0 - qh1)))) - 
    ((qh0 - qh1)*(qh0 + Power(ρh,2)))/Power(δqh + m*(qh0 - qh1),2) - 
    Log(δqh)/Power(m,2) + Log(δqh + m*(qh0 - qh1))/Power(m,2))/2

#### ENERGETIC TERM ####

function argGe(q0,δq,z0,z1)
    c = (√(1-q0)*z1 + √q0*z0) / √δq
    c > 0 && return 0.
    1/2 * min(c^2, 4.)    
end

function ∂c_argGe(q0,δq,z0,z1)
    c = (√(1-q0)*z1 + √q0*z0) / √δq
    c > 0 && return (0., 0.)
    c < -2 && return (0., 4.)
    (c, 1/2*c^2)    
end

function Ge(q0, δq, ρ, m)
    2∫D(z0->begin
        l = log(∫D(z1 -> exp(-m*argGe(q0, δq, z0, z1))))
        l * H(-√(ρ^2/(q0-ρ^2))*z0)  
        # l/2
    end)/m
end

# function Ge(q0, δq, ρ, m)
#     2∫D(z0->begin
#         l = log1p(∫D(z1 -> expm1(-m*argGe(q0, δq, z0, z1))))
#         l * H(-√(ρ^2/(q0-ρ^2))*z0)  
#         # l/2
#     end)/m
# end

# function Ge(q0, δq, ρ, m)
#     2∫D(z0->begin
#         l = log(∫Dexp(z1 -> -m*argGe(q0, δq, z0, z1)))
#         l * H(-√(ρ^2/(q0-ρ^2))*z0)  
#         # l/2
#     end)/m
# end

function ∂q0_Ge(q0, δq, ρ, m)
    2∫D(z0->begin
        num = ∫D(z1 -> begin
                dc, arg = ∂c_argGe(q0,δq,z0,z1)
                dc *= -m*(z0/(2Sqrt(q0)) - z1/(2.*Sqrt(1 - q0)))/Sqrt(δq)
                dc * exp(-m*arg)
            end)
        den = ∫D(z1 -> exp(-m*argGe(q0, δq, z0, z1)))

        dH = G(-√(ρ^2/(q0-ρ^2))*z0) * (-(Power(ρ,2)*z0)/
            (2.*Sqrt(Power(ρ,2)/(q0 - Power(ρ,2)))*Power(q0 - Power(ρ,2),2)))

        num/den * H(-√(ρ^2/(q0-ρ^2))*z0) + dH*log(den) #todo add derivatives
        # num/den/2
    end)/m
end

# function ∂q0_Ge(q0, δq, ρ, m)
#     2∫D(z0->begin
#         logden = log1p(∫D(z1 -> expm1(-m*argGe(q0, δq, z0, z1))))
#         num = ∫D(z1 -> begin
#                 dc, arg = ∂c_argGe(q0,δq,z0,z1)
#                 dc *= -m*(z0/(2Sqrt(q0)) - z1/(2.*Sqrt(1 - q0)))/Sqrt(δq)
#                 dc * exp(-m*arg - logden)
#             end)
#         # den = exp(logden)

#         dH = G(-√(ρ^2/(q0-ρ^2))*z0) * (-(Power(ρ,2)*z0)/
#             (2.*Sqrt(Power(ρ,2)/(q0 - Power(ρ,2)))*Power(q0 - Power(ρ,2),2)))

#         num * H(-√(ρ^2/(q0-ρ^2))*z0) + dH*logden #todo add derivatives
#         # num/den/2
#     end)/m
# end

# function ∂q0_Ge(q0, δq, ρ, m)
#     2∫D(z0->begin
#         num = ∫Dexp(z1 -> -m*argGe(q0, δq, z0, z1),

#                 dc, arg = ∂c_argGe(q0,δq,z0,z1)
#                 dc *= -m*(z0/(2Sqrt(q0)) - z1/(2.*Sqrt(1 - q0)))/Sqrt(δq)
#                 dc * exp(-m*arg)
#             end)
#         den = ∫Dexp(z1 -> -m*argGe(q0, δq, z0, z1))

#         dH = G(-√(ρ^2/(q0-ρ^2))*z0) * (-(Power(ρ,2)*z0)/
#             (2.*Sqrt(Power(ρ,2)/(q0 - Power(ρ,2)))*Power(q0 - Power(ρ,2),2)))

#         num/den * H(-√(ρ^2/(q0-ρ^2))*z0) + dH*log(den) #todo add derivatives
#         # num/den/2
#     end)/m
# end


function ∂δq_Ge(q0, δq, ρ, m)
    2∫D(z0->begin
        # logden = log1p(∫D(z1 -> expm1(-m*argGe(q0, δq, z0, z1))))
        
        num = ∫D(z1 -> begin
                dc, arg = ∂c_argGe(q0,δq,z0,z1)
                dc *= -m*(-(Sqrt(q0)*z0 + Sqrt(1 - q0)*z1)/(2.*Power(δq,1.5)))
                dc * exp(-m*arg)
            end)
        den = ∫D(z1 -> exp(-m*argGe(q0, δq, z0, z1)))
        # den = exp(logden)

        num/den * H(-√(ρ^2/(q0-ρ^2))*z0)  
        # num/den/2
    end)/m
end

function ∂ρ_Ge(q0, δq, ρ, m)
    #return 0.
    2∫D(z0->begin
        l = log(∫D(z1 -> exp(-m*argGe(q0, δq, z0, z1))))
        # l = log1p(∫D(z1 -> expm1(-m*argGe(q0, δq, z0, z1))))
        dH = G(-√(ρ^2/(q0-ρ^2))*z0) * ((q0*z0)/Power(q0 - Power(ρ,2),1.5))
        dH*l 
    end)/m
end

function ∂m_Ge(q0, δq, ρ, m)
    #return 0.
    I1 = 2∫D(z0->begin
        num = ∫D(z1 -> begin
                arg = argGe(q0,δq,z0,z1)
                (-arg) * exp(-m*arg)
            end)
        den = ∫D(z1 -> exp(-m*argGe(q0, δq, z0, z1)))
        num/den * H(-√(ρ^2/(q0-ρ^2))*z0)
    end)/m

    I2 = -1/m * Ge(q0, δq, ρ, m)
    I1 + I2 
end


# function ∂m_Ge(q0, δq, ρ, m)
#     #return 0.
#     I1 = 2∫D(z0->begin
#         num = ∫Dexp(z1 -> -m*argGe(q0,δq,z0,z1),
#                     z1 -> -argGe(q0,δq,z0,z1))
#         den = ∫Dexp(z1 -> -m*argGe(q0, δq, z0, z1))
#         num/den * H(-√(ρ^2/(q0-ρ^2))*z0)
#     end)/m

#     I2 = -1/m * Ge(q0, δq, ρ, m)
#     I1 + I2 
# end

# ∂q0_Ge(q0, δq, ρ, m) = deriv(Ge, 1, q0, δq, ρ, m) 
# ∂δq_Ge(q0, δq, ρ, m) = deriv(Ge, 2, q0, δq, ρ, m) 
# ∂ρ_Ge(q0, δq, ρ, m) = deriv(Ge, 3, q0, δq, ρ, m) 
# ∂m_Ge(q0, δq, ρ, m) = deriv(Ge, 4, q0, δq, ρ, m) 

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 δq qh0 qh1 δqh ρh m 
    @extract ep: α ρ
    @show Ge(q0,δq,ρ,m)
    Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) + Gs(qh0,qh1,δqh,ρh,m) + α*Ge(q0,δq,ρ,m)
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

#################  SADDLE POINT  ##################
fqh0(q0, δq, ρ, m, α) = -2/m * α * ∂q0_Ge(q0, δq, ρ, m)
fqh1(q0, δq, ρ, m, α) = 2α * ∂δq_Ge(q0, δq, ρ, m)
fρh(q0, δq, ρ, m, α) = α * ∂ρ_Ge(q0, δq, ρ, m)

fq0(qh0,qh1,δqh,ρh,m) = (qh0 + ρh^2) / (δqh + m*(qh0 - qh1))^2
fδq(qh0,qh1,δqh,ρh,m) = - m + 1/(δqh + m*(qh0 - qh1)) + (m*(qh0 + Power(ρh,2)))/Power(δqh + m*(qh0 - qh1),2)

fρ(qh0,qh1,δqh,ρh,m) = ρh/(δqh + m*(qh0 - qh1))

iρh(ρ,qh0,qh1,δqh,m) = (true, ρ*(δqh + m*(qh0 - qh1)))

function iδqh_fun(qh0, qh1, δqh, ρh, m)
    0.5 + (1/(δqh*m) - 1/(m*(δqh + m*(qh0 - qh1))) - 
    (qh0 + Power(ρh,2))/Power(δqh + m*(qh0 - qh1),2))/2.
end

function iδqh(qh0, qh1, δqh₀, ρh, m, atol=1e-10)
    ok, δqh, it, normf0 = findroot(δqh -> iδqh_fun(qh0, qh1, δqh, ρh, m), δqh₀, NewtonMethod(atol=atol))
    #ok, M, it, normf0 = findzero_interp(M->∂_Ge(5, Q, q0, q1, δq, M, x, K, avgξ, varξ, f′), M0, dx=0.1)

    ok || error("iδqh failed: iδqh=$(δqh), it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, δqh
end

function im_fun(op::OrderParams, ep::ExtParams, m)
    @extract op: q0 δq qh0 qh1 δqh ρh
    @extract ep: α ρ
    ∂m_Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) + ∂m_Gs(qh0,qh1,δqh,ρh,m) + α*∂m_Ge(q0,δq,ρ,m)
end

function im(op::OrderParams, ep::ExtParams, m₀, atol=1e-8)
    ok, m, it, normf0 = findroot(m -> im_fun(op, ep, m), m₀, NewtonMethod(atol=atol))
    ok || error("im failed: m=$m, it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, m
end

###############################


function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixm = false, fixρ = true)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    it = 0
    for it = 1:maxiters
        Δ = 0.0
        verb > 1 && println("it=$it")


        @update  op.qh0    fqh0       Δ ψ verb  op.q0 op.δq ep.ρ op.m ep.α
        @update  op.qh1    fqh1       Δ ψ verb  op.q0 op.δq ep.ρ op.m ep.α
        @updateI op.δqh ok   iδqh     Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        if fixρ
            @updateI op.ρh ok   iρh   Δ ψ verb  ep.ρ op.qh0 op.qh1 op.δqh op.m
        else
            @update  op.ρh  fρh       Δ ψ verb  op.q0 op.δq ep.ρ op.m ep.α
        end

        # fix_inequalities_hat!(op, ep)
        # fix_inequalities_nonhat!(op, ep)        

        @update op.q0   fq0       Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        @update op.δq   fδq       Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        end

        if !fixm ## && it%5==0
            @updateI op.m ok   im    Δ ψ verb  op ep op.m
        end

        verb > 1 && println(" Δ=$Δ\n")
        verb > 1 && println(all_therm_func(op, ep))

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end

    ok
end

function converge(;
        q0=0.1, δq=0.5,
        qh0=0., qh1=0., δqh=0.6,
        m = 1.00001, ρ=0, ρh=0,
        α=0.1,
        ϵ=1e-6, maxiters=100000, verb=2, ψ=0.,
        fixm = false, fixρ=true
    )
    op = OrderParams(q0,δq,qh0,qh1,δqh, ρh,m)
    ep = ExtParams(α, ρ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixm=fixm, fixρ=fixρ)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
    q0=0.1, δq=0.5,
    qh0=0., qh1=0., δqh=0.6, ρh=0,
    m=0.1, ρ=0.384312, α=1,
    ϵ=1e-6, maxiters=10000,verb=2, ψ=0.,
    kws...)

    op = OrderParams(q0,δq,qh0,qh1,δqh,ρh, first(m))
    ep = ExtParams(first(α), first(ρ))
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; ρ=ρ,α=α,m=m, kws...)
end


function span!(app::TheApp, op::OrderParams, ep::ExtParams;
        m=0,  α=1, ρ=1,
        resfile = "results.txt",
        fixm=false, fixρ=false)

    if !isfile(resfile)
        open(resfile, "w") do f
            allheadersshow(f, ExtParams, OrderParams, ThermFunc)
        end
    end

    results = []
    for m in m, α in α, ρ in ρ
        fixm && (op.m = m)
        fixρ && (ep.ρ = ρ)
        ep.α = α;

        if fixm
            ok = converge!(app, op, ep, fixm=true, fixρ=fixρ)
        else
            ok, m, it, normf0 = findroot(m -> begin
                                    println("# FINDROOT m=$m")
                                    op.m = m 
                                    ok = converge!(app, op, ep, fixm=true, fixρ=fixρ)
                                    println(op)
                                    println(all_therm_func(app, op, ep))
                                    im_fun(app, op, ep, m)
                                end, op.m, NewtonMethod(atol=app.ϵ, dx=100*app.ϵ))
            ok || error("im failed: m=$m, it=$it, normf0=$normf0")
            op.m = m  
        end
        tf = all_therm_func(app, op, ep)
        tf.Σ < -1e-7 && @warn("Sigma negative")
        push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))
        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(op), " ", plainshow(tf))
        end
        !ok && break
        app.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end

using Plots; plotlyjs(size=(1000,800))

function extrapolateβ(file)
    res = readdlm(file)

    β = res[:,2]
    @assert length(union(res[:,1])) == 1
    @assert length(union(res[:,3])) == 1
    α = res[1,1]
    ρ = res[1,3]

    model(x, p) = p[1] .+ p[2] .* x .+ p[3] .* x.^2
    p₀ = [0.5,0.5,0.5]
    q0 = curve_fit(model, 1./β, res[:,4], p₀).param[1]
    plot(1./β, res[:,4],label="q0")
    δq = curve_fit(model, 1 ./ β, (1-res[:,5]).*β, p₀).param[1]
    plot!(1 ./ β, (1-res[:,5]).*β, label="δq")
    qh0 = curve_fit(model, 1 ./ β, res[:,6]./β.^2, p₀).param[1]
    plot!(1 ./ β, res[:,6]./β.^2, label="qh0")
    qh1 = curve_fit(model, 1 ./ β, res[:,7]./β.^2, p₀).param[1]
    plot!(1 ./ β, res[:,7]./β.^2, label="qh1")
    δqh = curve_fit(model, 1 ./ β, (res[:,7].-res[:,8])./β, p₀).param[1]
    plot!(1 ./ β, (res[:,7].-res[:,8])./β, label="δqh")
    ρh = curve_fit(model, 1 ./ β, res[:,9] ./ β, p₀).param[1]
    plot!(1 ./ β, res[:,9] ./ β, label="ρh")
    m = curve_fit(model, 1 ./ β, res[:,10] .* β, p₀).param[1]
    plot!(1 ./ β, res[:,10] .* β, label="m")
    xlims!(0,0.1)
    ylims!(0,2)
    gui()
    OrderParams(q0, δq, qh0, qh1, δqh, ρh, m), ExtParams(α, ρ)
end

function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    ep = ExtParams(res[line,1:2]...)
    op = OrderParams(res[line,3:9]...)
    tf = ThermFunc(res[line,10:end]...)
    return ep, op, tf
end

end ## module
