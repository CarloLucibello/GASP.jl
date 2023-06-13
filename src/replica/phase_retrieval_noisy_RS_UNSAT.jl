module PhaseRetr

using LittleScienceTools.Roots
using QuadGK
using AutoGrad
using ExtractMacro
using Cubature
using Cuba
using IterTools: product
include("../common.jl")
# include("interpolation.jl")

###### INTEGRATION  ######
const ∞ = 12.0
const dx = 0.05

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞
const interval0 = (0:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        # isfinite(r) ? r : 0.0
    end, int..., abstol=1e-7, maxevals=10^7)[1]

## Cubature.jl
∫∫∫D(f, xmin::Vector, xmax::Vector) = hcubature(z->begin
            G(z[1])*G(z[2])*G(z[3])*f(z[1],z[2],z[3])
            # isfinite(r) ? r : 0.0
        end, xmin, xmax, abstol=1e-7)[1]


## Cuba.jl.
# ∫∫∫D(f, xmin::Vector, xmax::Vector) = cuhre((z,y)->begin
#             @. z = xmin + z*(xmax-xmin)
#             y[1] = G(z[1])*G(z[2])*G(z[3])*f(z[1],z[2],z[3])
#             # isfinite(r) ? r : 0.0
#         end, 3, 1,  abstol=1e-12)[1][1]*prod(xmax.-xmin)

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

# ∫d0(f, int=interval0) = quadgk(f,
#     int..., abstol=1e-6, maxevals=10^7)[1]

# Numerical Derivative
# Can take also directional derivative
# (tune the direction with i and δ).
function deriv(f::Function, i, x...; δ = 1e-3)
    x0 = deepcopy(x) |> collect
    x0[i] .-= δ
    f0 = f(x0...)
    x1 = deepcopy(x) |> collect
    x1[i] .+= δ
    f1 = f(x1...)
    return (f1-f0) / (2vecnorm(δ))
end

# deriv(f::Function, i::Integer, x...) = grad(f, i)(x...)


############### PARAMS ################

mutable struct OrderParams
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
Gi(δq,qh0,δqh,ρh,ρ) = (δqh - 2*ρ*ρh - δq*qh0)/2

#### ENTROPIC TERM ####

Gs(qh0,δqh,ρh) = 0.5*(Power(ρh,2) + qh0)/δqh

#### ENERGETIC TERM ####

fargGe(y, δq, h, u) = 1/2 * u^2 + 1/2 * (y - (u * √δq + h)^2)^2

function argGe(y, δq, h)
    ### findmin of 1/2 u^2 + 1/2 * (y - (u √δq + h)^2)^2
    a = 9 * √δq * h
    b = 2 * y * δq - 1
    c = 6*(-a + sqrt(complex(a^2 - 6 * b^3)))
    bc3 = b/c^(1/3)
    !isfinite(bc3) && (bc3 = zero(Complex)) # guard for b~0
    # @assert isfinite(bc3) "a=$a  b=$b  c=$c"
    c3 = c^(1/3)
    r1 = 1/δq * real((-a/9 -bc3 - c3/6))
    r2 = 1/δq * real((-a/9 + (1+√complex(-3))/2 * bc3 + (1-√complex(-3))/2 * c3/6))
    r3 = 1/δq * real((-a/9 + (1-√complex(-3))/2 * bc3 + (1+√complex(-3))/2 * c3/6))
    minimum(r->fargGe(y, δq, h, r), (r1, r2, r3))
end


# Qge(y,z0,ρ) = -(ρ^2*z0^2+y)/(2(1-ρ^2)) -0.5*log(2π*y*(1-ρ^2)) + logcosh(√y*ρ*z0/(1-ρ^2))

# function QgeΔ0(y, h, b)
#     s = -(h^2+y)/(2b) -0.5*log(2π*y*b) + logcosh(√y * h / b)
#     exp(s)
# end

# let  qfact = Dict{Float64, Interp}()

#     global function Ge(δq, ρ)
#         Δ = 1e-3
#         c = 4
#         if !haskey(qfact, Δ)
#             @time qfact[Δ] = Interp((y, h, b) -> begin
#                     ∫D(u -> begin
#                         exp(-(y - (h+b*u)^2)^2/(2Δ))
#                     end) / √(2π*Δ)
#                 end, (0:dx*∞/c:∞), -∞:dx*∞/c:∞, 0:dx/c:1)
#         end
#         Qge = qfact[Δ]

#         -∫d0(y->begin
#             ∫D(z0->begin
#                 Qge(y, ρ*z0, √(1-ρ^2)) * argGe(y, δq, z0)
#             end)
#         end)
#     end
# end #let

function Ge(δq, ρ, Δ)
    -∫∫∫D((u,ξ,z0)-> argGe(u^2 +√Δ*ξ, δq, ρ*u + √(1-ρ^2)*z0))
end



# function ∂δq_argGe(δq,z0)
#     c =  z0 / √δq
#     c > 0 && return 0.
#     c < -2 && return 0
#     -0.5*z0^2 / (δq)^2
# end

# function ∂δq_Ge(δq, ρ)
#     -2∫D(z0->begin
#         l = ∂δq_argGe(δq, z0)
#         l * H(-√(ρ^2/(1-ρ^2))*z0)  # teacher-student
#         # l/2   # student
#     end)
# end

# function ∂ρ_Ge(δq, ρ)
#     -2∫D(z0->begin
#         l = argGe(δq, z0)
#         dH = G(-√(ρ^2/(1-ρ^2))*z0) * (z0/Power(1 - Power(ρ,2),1.5))
#         l * dH  # teacher-student
#     end)
# end

∂δq_Ge(δq, ρ, Δ) = deriv(Ge, 1, δq, ρ, Δ)
∂ρ_Ge(δq, ρ, Δ) = deriv(Ge, 2, δq, ρ, Δ)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: δq qh0 δqh ρh
    @extract ep: α ρ Δ
    Gi(δq,qh0,δqh,ρh,ρ) + Gs(qh0,δqh,ρh) + α*Ge(δq,ρ, Δ)
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
fqh0(δq, ρ, α, Δ) = 2α * ∂δq_Ge(δq, ρ, Δ)
fρh(δq, ρ, α, Δ) = α * ∂ρ_Ge(δq, ρ, Δ)

fδq(qh0,δqh,ρh) = 1/δqh
fρ(qh0,δqh,ρh) = ρh/δqh

iρh(ρ,qh0,δqh) = (true, ρ*δqh)

function iδqh(qh0, δqh, ρh)
    (true, sqrt(qh0 + ρh^2))
end

###############################


function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixρ = true)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    it = 0
    for it = 1:maxiters
        tic()
        Δ = 0.0
        verb > 1 && println("it=$it")

        @update  op.qh0    fqh0       Δ ψ verb  op.δq ep.ρ ep.α ep.Δ
        @updateI op.δqh ok   iδqh     Δ ψ verb  op.qh0 op.δqh op.ρh
        if fixρ
            @updateI op.ρh ok   iρh   Δ ψ verb  ep.ρ op.qh0 op.δqh
        else
            @update  op.ρh  fρh       Δ ψ verb  op.δq ep.ρ ep.α ep.Δ
        end

        # fix_inequalities_hat!(op, ep)
        # fix_inequalities_nonhat!(op, ep)

        @update op.δq   fδq       Δ ψ verb  op.qh0 op.δqh op.ρh
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op.qh0 op.δqh op.ρh
        end


        verb > 1 && println(" Δ=$Δ\n")
        verb > 2 && it%5==0 && (println(op);println(all_therm_func(op, ep)))

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        toc()
        ok && break
    end

    ok
end

function converge(;
        δq=0.5,
        qh0=0., δqh=0.6,
        ρ=0, ρh=0,
        α=0.1, Δ=1e-3,
        ϵ=1e-6, maxiters=100000, verb=3, ψ=0.,
        fixm = false, fixρ=true
    )
    op = OrderParams(δq,qh0,δqh,ρh)
    ep = ExtParams(α, ρ, Δ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixρ=fixρ)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
    δq=0.3188,
    qh0=0.36889,δqh=0.36889, ρh=0.56421,
    ρ=0.384312, α=1, Δ = 0.,
    ϵ=1e-6, maxiters=10000,verb=3, ψ=0.,
    kws...)

    op = OrderParams(δq,qh0,δqh,ρh)
    ep = ExtParams(first(α), first(ρ), Δ)
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; ρ=ρ,α=α, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
    α=1, ρ=1,
    resfile = "results.txt",
    fixm=false, fixρ=false, mlessthan1=false)

    if !isfile(resfile)
        open(resfile, "w") do f
            allheadersshow(f, ExtParams, ThermFunc, OrderParams)
        end
    end

    results = []
    for α in α, ρ in ρ
        fixρ && (ep.ρ = ρ)
        ep.α = α;

        ok = converge!(op, ep, pars; fixρ=fixρ)
        tf = all_therm_func(op, ep)
        push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))
        if ok
            open(resfile, "a") do rf
                println(rf, plainshow(ep), " ", plainshow(tf), plainshow(op))
            end
        end
        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end

function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    ep = ExtParams(res[line,1:2]...)
    op = OrderParams(res[line,4:end]...)
    tf = ThermFunc(res[line,3])
    return ep, op, tf
end

end ## module
