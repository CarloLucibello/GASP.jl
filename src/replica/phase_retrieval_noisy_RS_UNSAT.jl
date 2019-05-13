module PhaseRetr

using LittleScienceTools.Roots
using QuadGK
using AutoGrad
using ExtractMacro
using Cubature
# using Cuba
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

∫∫D(f, xmin::Vector, xmax::Vector) = hcubature(z->begin
            G(z[1])*G(z[2])*f(z[1],z[2])
            # isfinite(r) ? r : 0.0
        end, xmin, xmax, abstol=1e-7)[1]

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

function ∫∫D(f) 
    ints = [(interval[i],interval[i+1]) for i=1:length(interval)-1]
    intprods = product(ints, ints)
    # @show collect(intprods)
    sum(ip-> begin
            xmin = [ip[1][1],ip[2][1]]
            xmax = [ip[1][2],ip[2][2]]
            ∫∫D(f, xmin, xmax)
        end, intprods)
end

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
    stab::Float64 # local stability RS solution
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
    (q0*δqh - 2*ρ*ρh - δq*qh0)/2
end

#### ENTROPIC TERM ####

function Gs(op, ep)    
    @extract op: qh0 δqh ρh
    0.5*(Power(ρh,2) + qh0)/δqh
end

#### ENERGETIC TERM ####

fargGe(y, δq, h, u) = 1/2 * u^2 + 1/2 * (y - (u * √δq + h)^2)^2

function argGe(y, δq, h)
    a = 9 * √δq * h
    b = 2 * y * δq - 1
    c = (6*(-a + sqrt(complex(a^2 - 6 * b^3))))^(1/3)
    bc = b/c
    !isfinite(bc) && (bc = zero(Complex128)) # guard for b~0  
    # @assert isfinite(bc3) "a=$a  b=$b  c=$c"
    u1 = 1/δq * real((-a/9 -bc - c/6))
    u2 = 1/δq * real((-a/9 + (1+√complex(-3))/2 * bc + (1-√complex(-3))/2 * c/6))
    u3 = 1/δq * real((-a/9 + (1-√complex(-3))/2 * bc + (1+√complex(-3))/2 * c/6))
    minimum(u->fargGe(y, δq, h, u), (u1, u2, u3))
end

function argGe_min(y, δq, h)
    ### findmin of 1/2 u^2 + 1/2 * (y - (u √δq + h)^2)^2
    a = 9 * √δq * h
    b = 2 * y * δq - 1
    c = (6*(-a + sqrt(complex(a^2 - 6 * b^3))))^(1/3)
    bc = b/c
    !isfinite(bc) && (bc = zero(Complex128)) # guard for b~0  
    # @assert isfinite(bc3) "a=$a  b=$b  c=$c"
    u1 = 1/δq * real((-a/9 -bc - c/6))
    u2 = 1/δq * real((-a/9 + (1+√complex(-3))/2 * bc + (1-√complex(-3))/2 * c/6))
    u3 = 1/δq * real((-a/9 + (1-√complex(-3))/2 * bc + (1+√complex(-3))/2 * c/6))

    roots = (u1,u2,u3)
    # mins = fargGe.(y, δq, h, roots)
    m, am_ind = findmin(map(u->fargGe(y, δq, h, u), roots))
    am = roots[am_ind]
    # am_ind = indmin(mins)
    # am = roots[am_ind]
    # m  = mins[am_ind]
    # @show y  am*√δq+h  m 
    return am, m
    # roots[1], mins[1]
    # u1, u2
    # 1., 1.
end

fyp_(q0, δq, z0, u) = u * √(δq) + z0 * √(q0)
fargGe_(y, yp, u) = 1/2 * u^2 + 1/2 * (y^2 - yp^2)^2

function fargGe_min(y, q0, δq, z0; argmin=false)
    ### findmin of 1/2 u^2 + 1/2 * (y^2 - (u √δq + z0 √q0)^2)^2
    a = 9 * √(δq) * √(q0) * z0
    b = 2 * y^2 * δq - 1
    c = 6*(-a + sqrt(complex(a^2 - 6 * b^3)))
    c3 = c^(1/3)
    bc3 = b / c3

    u1 = 1/δq * real((-a/9 -bc3 - c3/6))
    u2 = 1/δq * real((-a/9 + (1+√complex(-3))/2 * bc3 + (1-√complex(-3))/2 * c3/6))
    u3 = 1/δq * real((-a/9 + (1-√complex(-3))/2 * bc3 + (1+√complex(-3))/2 * c3/6))

    roots = [u1,u2,u3]
    if argmin
        m, am_ind = findmin(map(u->fargGe_(y, fyp_(q0, δq, z0, u), u), roots))
        am = roots[am_ind]
        yp = fyp_(q0, δq, z0, am)
        return am, m, yp
    end
    minimum(r->fargGe_(y, fyp_(q0, δq, z0, r), r), roots)
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

function Ge(op, ep)
    @extract op: q0 δq
    @extract ep: ρ Δ
    if Δ > 0
        -∫∫∫D((u,ξ,z0)-> argGe(u^2 +√Δ*ξ, δq, ρ*u + √(q0-ρ^2)*z0))
    else
        -∫∫D((u,z0)-> argGe(u^2, δq, ρ*u + √(q0-ρ^2)*z0))
    end    
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

∂q0_Ge(op,ep) = deriv_(Ge, 1, op, ep)
∂δq_Ge(op,ep) = deriv_(Ge, 2, op, ep)
∂ρ_Ge(op,ep) = deriv_(Ge, 2, op, ep, arg=2)

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    Gi(op,ep) + Gs(op,ep) + ep.α*Ge(op, ep)
end

function stability(op, ep)
    @extract op: q0 δq δqh
    @extract ep: ρ α Δ
    # return 0.

    γS = 1/δqh^2 
    # γE = < (ℓ'' / (1+δq*ℓ''))^2 >
    if Δ > 0 
        γE = ∫∫∫D((u0,ξ,z0) -> begin
                h = ρ*u0 + √(q0 - ρ^2)*z0
                y = u0^2 + √Δ*ξ        
                umin, fmin = argGe_min(y, δq, h)
                l2 = 6*(umin*δq + h)^2 - 2y 
                (l2 / (1 + δq*l2))^2
            end)
    else
        γE = ∫∫D((u0,z0)->begin
                h = ρ*u0 + √(q0 - ρ^2)*z0        
                y = u0^2        
                umin, fmin = argGe_min(y, δq, h)
                l2 = 6*(umin*√δq + h)^2 - 2y 
                l2den = (1 + δq*l2)
                l2den <= 0 && (l2den=1e-8)
                # der = umin + 2*√δq*(umin*√δq + h)*((umin*√δq + h)^2 - y)
                # @show y δq h umin
                # @assert der ≈ 0 "der=$der"
                # @show l2den
                l3 = (l2 / l2den)^2
                # @show l2 l2den l3 
                l3
                # # @assert l2 >= 0
                # (l2 / (1 + δq*l2))^2
            end)
    end

    return 1 - α*γS*γE  # α*γS*γE < 1 => the solution is locally stable
end

function all_therm_func(op::OrderParams, ep::ExtParams)
    ϕ = free_entropy(op, ep)
    stab = stability(op, ep)
    return ThermFunc(ϕ, stab)
end

#################  SADDLE POINT  ##################

fδqh(op, ep) = -2ep.α * ∂q0_Ge(op, ep)
fqh0(op, ep) = 2ep.α * ∂δq_Ge(op, ep)
fρh(op, ep) = ep.α * ∂ρ_Ge(op, ep)

fq0(op) = (op.ρh^2 + op.qh0) / op.δqh^2
fδq(op) = 1 / op.δqh
fρ(op) = op.ρh / op.δqh

iρh(op, ep) = (true, ep.ρ*op.δqh)

function iδqh(op)
    (true, sqrt(op.qh0 + op.ρh^2))
end


###############################

function fix_inequalities!(op, ep)
    if op.q0 < ep.ρ^2
        op.q0 = sqrt(ep.ρ) + 1e-3
    end
end

function converge!(op::OrderParams, ep::ExtParams, pars::Params;
        fixρ=true, fixnorm=true, extrap=-1)
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
            @updateI op.ρh oki   iρh   Δ ψ verb  op ep
            ok &= oki
        else
            @update  op.ρh  fρh       Δ ψ verb  op ep
        end

        @update op.δq   fδq       Δ ψ verb  op
        if !fixnorm
            @update op.q0   fq0     Δ ψ verb  op
        end
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op
        end


        verb > 1 && println(" Δ=$Δ\n")
        verb > 2 && it%5==0 && (println(ep);println(all_therm_func(op, ep));println(op))

        @assert isfinite(Δ)
        ok &= Δ < ϵ
        ok && break

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
        q0=0.4, δq=0.5,
        qh0=0., δqh=0.6,
        ρ=0, ρh=0,
        α=0.1, Δ=0.,
        ϵ=1e-6, maxiters=100000, verb=3, ψ=0.,
        fixρ=true, fixnorm=false, extrap=-1
    )
    op = OrderParams(q0,δq,qh0,δqh,ρh)
    ep = ExtParams(α, ρ, Δ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixρ=fixρ, fixnorm=fixnorm,extrap=extrap)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
    q0 = 0.4, δq=0.3188,
    qh0=0.36889,δqh=0.36889, ρh=0.56421,
    ρ=0.384312, α=1, Δ = 0.,
    ϵ=1e-6, maxiters=10000,verb=3, ψ=0.,
    kws...)

    op = OrderParams(q0,δq,qh0,δqh,ρh)
    ep = ExtParams(first(α), first(ρ), Δ)
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
        !ok && break
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end

function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    ep = ExtParams(res[line,1:3]...)
    op = OrderParams(res[line,5:end]...)
    tf = ThermFunc(res[line,4])
    return ep, op, tf
end

end ## module
