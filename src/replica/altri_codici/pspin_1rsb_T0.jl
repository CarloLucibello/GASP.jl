module PSpin

using LittleScienceTools.Roots
include("../../common.jl")

############### PARAMS

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
    p::Int
    k::Int # field exponent
    r::Float64 # field force
    ρ::Float64 # overlap with north pole
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

function shortshow(io::IO, x)
    T = typeof(x)
    print(io, T.name.name, "(", join([string(f, "=", getfield(x, f)) for f in fieldnames(T)], ","), ")")
end

function plainshow(x)
    T = typeof(x)
    join([getfield(x, f) for f in fieldnames(T)], " ")
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
Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) = (δqh - 2*ρ*ρh - δq*qh1 + q0*qh0*m - qh1*m)/2

∂m_Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) = (q0*qh0 - qh1) / 2

#### ENTROPIC TERM ####

Gs(qh0,qh1,δqh,ρh,m) = 0.5*((Power(ρh,2) + qh0)/(δqh + (qh0 - qh1)*m) + Log(δqh)/m - Log(δqh + (qh0 - qh1)*m)/m)

∂m_Gs(qh0,qh1,δqh,ρh,m) =  (-((qh0 - qh1)/(m*(δqh + m*(qh0 - qh1)))) - 
    ((qh0 - qh1)*(qh0 + Power(ρh,2)))/Power(δqh + m*(qh0 - qh1),2) - 
    Log(δqh)/Power(m,2) + Log(δqh + m*(qh0 - qh1))/Power(m,2))/2

#### ENERGETIC TERM ####

function Ge(q0, δq, m, p, k, r, ρ)
    (p*δq + m - q0^p*m)/4 + r*ρ^k/k
end

∂q0_Ge(q0, δq, m, p, k, r,ρ) = -p*q0^(p-1)*m/4
∂δq_Ge(q0, δq, m, p, k, r,ρ) = p/4
∂ρ_Ge(q0, δq, m, p, k, r,ρ) = r*ρ^(k-1) 
∂m_Ge(q0,δq,m,p,k,r,ρ) = (1-q0^p)/4 

############ Thermodynamic functions

function free_entropy(op::OrderParams, ep::ExtParams)
    @extract op: q0 δq qh0 qh1 δqh ρh m 
    @extract ep: p k r ρ
    Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) + Gs(qh0,qh1,δqh,ρh,m) + Ge(q0,δq,m,p,k,r,ρ)
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
fqh0(q0,δq,m,p,k,r,ρ) = -2/m * ∂q0_Ge(q0,δq,m,p,k,r,ρ)
fqh1(q0,δq,m,p,k,r,ρ) = 2∂δq_Ge(q0,δq,m,p,k,r,ρ)
fρh(q0,δq,m,p,k,r,ρ) = ∂ρ_Ge(q0,δq,m,p,k,r,ρ)

fq0(qh0,qh1,δqh,ρh,m) = (qh0 + ρh^2) / (δqh + m*(qh0 - qh1)^2)
fδq(qh0,qh1,δqh,ρh,m) = - m + 1/(δqh + m*(qh0 - qh1)) + (m*(qh0 + Power(ρh,2)))/Power(δqh + m*(qh0 - qh1),2)

fρ(qh0,qh1,δqh,ρh,m) = ρh/(δqh + m*(qh0 - qh1))

iρh(ρ,qh0,qh1,δqh,m) = (true, ρ*(δqh + m*(qh0 - qh1)))

function iδqh_fun(qh0, qh1, δqh, ρh, m)
    return (0.5 + (1/(δqh*m) - 1/(m*(δqh + m*(qh0 - qh1))) - 
    (qh0 + Power(ρh,2))/Power(δqh + m*(qh0 - qh1),2))/2.)
end

function iδqh(qh0, qh1, δqh₀, ρh, m, atol=1e-7)
    ok, δqh, it, normf0 = findroot(δqh -> iδqh_fun(qh0, qh1, δqh, ρh, m), δqh₀, NewtonMethod(atol=atol))
    #ok, M, it, normf0 = findzero_interp(M->∂_Ge(5, Q, q0, q1, δq, M, x, K, avgξ, varξ, f′), M0, dx=0.1)

    ok || @warn("iδqh failed: iδqh=$(δqh), it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    return ok, δqh
end

function im_fun(op::OrderParams, ep::ExtParams, m)
    @extract op: q0 δq qh0 qh1 δqh ρh
    @extract ep: p k r ρ
    ∂m_Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) + ∂m_Gs(qh0,qh1,δqh,ρh,m) + ∂m_Ge(q0,δq,m,p,k,r,ρ)
end

function im(op::OrderParams, ep::ExtParams, m₀, atol=1e-7)
    ok, m, it, normf0 = findroot(m -> im_fun(op, ep, m), m₀, NewtonMethod(atol=atol))
    ok || @warn("im failed: m=$m, it=$it, normf0=$normf0")
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
        @update  op.qh0    fqh0       Δ ψ verb  op.q0 op.δq op.m ep.p ep.k ep.r ep.ρ
        @update  op.qh1    fqh1       Δ ψ verb  op.q0 op.δq op.m ep.p ep.k ep.r ep.ρ
        @updateI op.δqh ok   iδqh     Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m  ϵ/10
        if fixρ
            @updateI op.ρh ok   iρh   Δ ψ verb  ep.ρ op.qh0 op.qh1 op.δqh op.m
        else
            @update  op.ρh  fρh       Δ ψ verb  op.q0 op.δq op.m ep.p ep.k ep.r ep.ρ
        end

        @update op.q0   fq0       Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        @update op.δq   fδq       Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        end

        if !fixm
            @updateI op.m ok   im    Δ ψ verb  op ep op.m  ϵ/10
        end

        verb > 1 && println(" Δ=$Δ\n")

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end

end

function converge(;
        q0=0.1, δq=0.5,
        qh0=0., qh1=0., δqh=0.6,
        m = 1.00001, ρ=0, ρh=0,
        p=3,k=3,r=0,
        ϵ=1e-6, maxiters=100000, verb=2, ψ=0.,
        fixm = false, fixρ=false
    )
    op = OrderParams(q0,δq,qh0,qh1,δqh,ρh,m)
    ep = ExtParams(p,k,r,ρ)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixm=fixm, fixρ=fixρ)
    tf = all_therm_func(op, ep)
    return op, ep, pars, tf
end


# function span(; lstα = [0.6], lstp = [0.6], op = nothing,
#                 ϵ = 1e-5, ψ = 0.1, maxiters = 500, verb = 4,
#                 resfile = nothing, updatem = true)

#     default_resfile = "results_phase_retrieval_zeroT_SAT.txt"
#     resfile == nothing && (resfile = default_resfile)

#     pars = Params(ϵ, ψ, maxiters, verb)
#     results = Any[]

#     lockfile = "reslock.tmp"

#     op == nothing && (op = initialize_op())
#     ep = ExtParams(lstα[1], lstp[1])
#     println(op)

#     for α in lstα, p in lstp
#         ep.α = α
#         ep.p = p

#         function seek!(m)
#             op.m = m
#             converge!(op, ep, pars, updatem = false)
#             return fmz(ep.α, op.m, op.q00, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2)
#         end

#         if !updatem
#             ok, op.m, it, norm = findroot(seek!, op.m, NewtonMethod(atol=ϵ))
#         else
#             converge!(op, ep, pars, updatem = true)
#         end

#         tf = ThermFunc(free_entropy(op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2))

#         push!(results, (ok, deepcopy(op), deepcopy(ep), deepcopy(tf)))
#         verb > 0 && println(tf)
#         println("\n#####  NEW ITER  ###############\n")
#         if ok
#             exclusive(lockfile) do
#                 open(resfile, "a") do rf
#                     println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
#                 end
#             end
#         end
#         ok || break
#     end
#     return results, op
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
