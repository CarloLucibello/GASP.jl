module PhaseRetr

using LittleScienceTools.Roots
using OffsetArrays
using TimerOutputs
using QuadGK
using FastGaussQuadrature
include("../common.jl")

ENABLE_TIMINGS = true

macro t(exprs...)
    if ENABLE_TIMINGS
        return :(@timeit($(esc.(exprs)...)))
    else
        return esc(exprs[end])
    end
end

###### INTEGRATION  ######
const ∞ = 12.0
const dx = 0.2

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-5, maxevals=10^7)[1]

# let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
#     global gw
#     gw(n::Int) = get!(s, n) do
#         (x,w) = gausshermite(n)
#         return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
#     end
# end

# function ∫D(f; n=400)
#     (xs, ws) = gw(n)
#     s = 0.0
#     for (x,w) in zip(xs, ws)
#         y = f(x)
#         s += w  * ifelse(isfinite(y), y, 0.0)
#     end
#     return s
# end

# Kahan summation
immutable AccumulaTrick
    sum::Float64
    r::Float64
end
AccumulaTrick() = AccumulaTrick(0.,0.)

function Base.:+(a::AccumulaTrick, x)
    if abs(a.sum) < abs(x)
        return AccumulaTrick(x, a.r) + a.sum 
        a.sum, x = x, a.sum
    end
    y = x - a.r
    z = a.sum + y
    AccumulaTrick(z, (z - a.sum) - y)
end

value(a::AccumulaTrick) = a.sum + a.r
value(a) = a

function deriv(f::Function, i, x...; δ = 1e-7)
    x1 = deepcopy(x) |> collect
    x1[i] .+= δ
    f0 = f(x...)
    f1 = f(x1...)
    return (f1-f0) / vecnorm(δ)
end

# deriv(f::Function, i::Integer, x...) = grad(f, i)(x...)

###### Array Types ########

const SymVec = OffsetVector{Float64}

symvec(L::Int) = (a=SymVec(-L:L); a .= 0; a)

const QVec{T} = OffsetVector{T}

function qvec(r::UnitRange, x0::T) where T
    a = QVec{T}(r)
    for k in r
        a[k] .= x0
    end
    a
end

##
mutable struct ExtVec
    v::SymVec
    L::Int
    a_left::Float64
    b_left::Float64
    a_right::Float64
    b_right::Float64
end

function ExtVec(L::Int = 0.)
    v = symvec(L)
    ExtVec(v, L, 0.,0.,0.,0.)
end

@inline Base.setindex!(v::ExtVec, x, i) = setindex!(v.v, x, i)
@inline Base.getindex(v::ExtVec, i) = i < -v.L ? v.a_left + v.b_left*i :
                         i > v.L ? v.a_right+v.b_right*i : getindex(v.v, i)

function extend_left!(v::ExtVec)
    L=v.L
    a=v[-L+1]-v[-L]
    b= v[-L]+L*a
    extend_left!(v, b, a)
end
function extend_right!(v::ExtVec)
    L = v.L
    a = v[L]-v[L-1]
    b = v[L]-L*a
    extend_right!(v, b, a)
end
extend_left!(v::ExtVec, a, b) = (v.a_left=a; v.b_left=b)
extend_right!(v::ExtVec, a, b) = (v.a_right=a; v.b_right=b)
extend_linear!(v) = (extend_left!(v); extend_right!(v))

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

mutable struct TheApp
    RES::Int # 1/a
    LIMIT::Int # width of Ge
    WIDTH::Int # kernel size
    a::Float64 # lattice spacing

    kernel::SymVec
    GEs::QVec{ExtVec} # Ge[k] = Ge at level k of the ultrametric tree. k=0,..,K 
    pH::QVec{ExtVec} # fields distributions
    Ge::Float64    # final value of Ge

    ϵ::Float64 # stop criterium
    ψ::Float64 # dumping
    maxiters::Int
    verb::Int

    function TheApp(RES, limit, width, ϵ, ψ, maxiters, verb)
        LIMIT = limit*RES
        WIDTH = width*RES
        kernel = symvec(WIDTH)
        K = 1
        GEs = qvec(0:1, ExtVec(LIMIT))
        pH = qvec(0:1, ExtVec(LIMIT))
        Ge = 0.
        new(RES, LIMIT, WIDTH, 1/RES,        
            kernel, GEs, pH, Ge,
            ϵ, ψ, maxiters, verb)
    end
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
Sqrt(x) = sqrt(x)

#### INTERACTION TERM ####
Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) = (q1*δqh - 2*ρ*ρh - δq*qh1 + q0*qh0*m - qi*qh1*m)/2

∂m_Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) = (q0*qh0 - qh1*q1) / 2

#### ENTROPIC TERM ####

Gs(qh0,qh1,δqh,ρh,m) = 0.5*((Power(ρh,2) + qh0)/(δqh + (qh0 - qh1)*m) + Log(δqh)/m - Log(δqh + (qh0 - qh1)*m)/m)

∂m_Gs(qh0,qh1,δqh,ρh,m) =  (-((qh0 - qh1)/(m*(δqh + m*(qh0 - qh1)))) - 
    ((qh0 - qh1)*(qh0 + Power(ρh,2)))/Power(δqh + m*(qh0 - qh1),2) - 
    Log(δqh)/Power(m,2) + Log(δqh + m*(qh0 - qh1))/Power(m,2))/2

#### ENERGETIC TERM ####

fargGe(y, δq, h, u) = 1/2 * u^2 + 1/2 * (y - (u * √δq + h)^2)^2

function argGe(y, δq, h)
    ### findmin of 1/2 u^2 + 1/2 * (y - (u √δq + h)^2)^2
    a = 9 * √δq * h
    b = 2 * y * δq - 1
    c = 6*(-a + sqrt(complex(a^2 - 6 * b^3)))
    bc3 = b/c^(1/3)
    !isfinite(bc3) && (bc3 = zero(Complex)) # guard for b~0  
    c3 = c^(1/3)
    r1 = 1/δq * real((-a/9 -bc3 - c3/6))
    r2 = 1/δq * real((-a/9 + (1+√complex(-3))/2 * bc3 + (1-√complex(-3))/2 * c3/6))
    r3 = 1/δq * real((-a/9 + (1-√complex(-3))/2 * bc3 + (1+√complex(-3))/2 * c3/6))
    minimum(r->fargGe(y, δq, h, r), (r1, r2, r3))
end

@timeit function firststepGe!(gK, theapp, δq, ρ, t)
    @extract theapp: a LIMIT WIDTH
    
    fK(h) = -argGe(t^2, δq, ρ*t + h)
    g = gK.v #symmetric vector
    @unsafe for i=-LIMIT:LIMIT
        h = a * i
        g[i] = fK(h)
    end
    extend_linear!(gK)
end

function init_kernel!(kernel, a, WIDTH, dQ)
    if dQ > a^2/100     #protection for small deltaQ
        #TODO: add @unsafe
        s = AccumulaTrick()
        @unsafe for i=-WIDTH:WIDTH
            kernel[i] = exp(-(a*i)^2 *(0.5/dQ))
            s += kernel[i]
        end
        kernel ./= value(s)
    else
        kernel .= 0
        kernel[0] = 1
    end
    kernel
end

function onestepGe_inner!(gnew, expg, kernel, LIMIT, WIDTH, m)
    for i=-LIMIT:LIMIT
        # s = AccumulaTrick() # x10 slower
        s = 0. 
        @unsafe for k=-WIDTH:WIDTH
            s += expg[i+k]*kernel[k]
        end
        gnew[i] = log(value(s)) / m
    end
end

@timeit function onestepGe!(gnew, theapp::TheApp, g, m, dQ)
    @extract theapp: kernel LIMIT WIDTH a
    expg = symvec(LIMIT+WIDTH)
    for i=-(LIMIT+WIDTH):LIMIT+WIDTH
        expg[i] = exp(m*g[i])
        !isfinite(expg[i]) && (expg[i] = 0) # protection from overflow
    end
    init_kernel!(kernel, a, WIDTH, dQ)
    # need a function barrier for performace (x10)
    # Why? investigate type instabilities
    onestepGe_inner!(gnew, expg, kernel, LIMIT, WIDTH, m)
    extend_linear!(gnew)
end

@timeit function laststepGe(theapp::TheApp, g, dQ)
    @extract theapp: kernel LIMIT WIDTH a
    init_kernel!(kernel, a, WIDTH, dQ)
    # c = -ρ * √(1/(q0*(q0-ρ^2))) * a
    s = AccumulaTrick()
    for k=-WIDTH:WIDTH
        s += g[k] * kernel[k]
    end
    value(s)
end

function computeGe!(theapp::TheApp, op::OrderParams, ep::ExtParams)
    @extract theapp: GEs WIDTH a
    @extract op: m q0 δq 
    @extract ep: ρ

    # s = AccumulaTrick()
    # tkern = init_kernel!(symvec(WIDTH), a, WIDTH, 1)
    # for i=-WIDTH:WIDTH
    #     firststepGe!(GEs[1], theapp, δq, ρ, a*i)
    #     onestepGe!(GEs[0], theapp, GEs[1], m, 1 - q0)
    #     s += laststepGe(theapp, GEs[0], q0 - ρ^2) * tkern[i]
    # end
    # theapp.Ge = value(s)
    theapp.Ge = ∫D(t -> begin
            firststepGe!(GEs[1], theapp, δq, ρ, t)
            onestepGe!(GEs[0], theapp, GEs[1], m, 1 - q0)
            laststepGe(theapp, GEs[0], q0 - ρ^2)
        end)
    
    theapp.Ge
end

M2G(g::ExtVec, i::Int, a::Float64) = ((g[i+1] - g[i-1])/(2.*a))^2

function fqh0_loop!(p, g, kernel, WIDTH, a)
    s = AccumulaTrick()
    for i=-WIDTH:WIDTH
        p[i] = kernel[i]
        s += M2G(g, i, a) * p[i]
        # @assert isfinite(value(qSomma))
    end
    value(s)
end

function fqh0!(app::TheApp, op, ep)
    @extract app: G=GEs P=pH kernel a WIDTH
    @extract op: m q0 δq
    @extract ep: ρ α

    # STEP k=0
    qh0 = α*∫D(t -> begin
            firststepGe!(G[1], app, δq, ρ, t)
            onestepGe!(G[0], app, G[1], m, 1 - q0)
            init_kernel!(kernel, a, WIDTH, q0 - ρ^2)
            fqh0_loop!(P[0], G[0], kernel, WIDTH, a)
        end)
    return qh0

    # # STEPS k>0
    # for k=1:K+1
    #     initKernel!(theapp, q[k] - q[k-1])
    #     g = G[k-1]
    #     p = P[k-1]
    #     for i=-(LIMIT+WIDTH) : LIMIT+WIDTH
    #         scra[i] = p[i] * exp(-m[k] * g[i])
    #         !isfinite(scra[i]) && (scra[i] = 0)
    #         # @assert isfinite(scra[i]) "isfinite(scra[i]) k=$k i=$i $(scra[i]) g[i]=$(g[i]) p[i]=$(p[i])"
    #     end
    #     qSomma = AccumulaTrick()
    #     @inbounds for i=-LIMIT:LIMIT
    #         somma = AccumulaTrick()
    #         for j=-WIDTH:WIDTH
    #             somma += scra[i-j]*kernel[j]
    #             # @assert isfinite(kernel[j]) "isfinite(kernel[j]) k=$k i=$i j=$j $(kernel[j])"
    #             # @assert isfinite(scra[i-j]) "isfinite(scra[i-j]) k=$k i=$i j=$j $(scra[i-j])"
    #             # @assert isfinite(value(somma)) "isfinite(value(somma)) k=$k i=$i j=$j $(value(somma))"
    #         end
    #         P[k][i] = value(somma) * exp(+m[k] * G[k][i])
    #         qSomma += M2G(G[k], i, a) * P[k][i]
    #     end
    #     qnew[k] = value(qSomma)
    # end
    # qnew[K+1] = qnew_t == :hat ? 0 : 1

    # if model == :perceptron && qnew_t == :hat
    #     for k=0:K+1
    #         qnew[k] *= α
    #     end
    # end
    # if model == :pspin && qnew_t == :nonhat
    #     (β == Inf) && (qnew[K] = 1)
    # end
end

function ∂δq_Ge(theapp::TheApp, op::OrderParams, ep::ExtParams)
    δ = 1e-5
    op.δq += δ
    f1 = computeGe!(theapp, op, ep)
    op.δq -= δ
    f0 = computeGe!(theapp, op, ep)
    (f1-f0) / δ
end

function ∂m_Ge(theapp::TheApp, op::OrderParams, ep::ExtParams)
    δ = 1e-5
    op.m += δ
    f1 = computeGe!(theapp, op, ep)
    op.m -= δ
    f0 = computeGe!(theapp, op, ep)
    (f1-f0) / δ
end

############ Thermodynamic functions

function free_entropy(app, op::OrderParams, ep::ExtParams)
    @extract op: q0 δq qh0 qh1 δqh ρh m 
    @extract ep: α ρ
    Ge = computeGe!(app, op, ep)
    # @show Ge
    Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) + Gs(qh0,qh1,δqh,ρh,m) + α*Ge
end

## Thermodinamic functions
# The energy of the pure states selected by x
# E = -∂(m*ϕ)/∂m
# if working at fixed x. If x is optimized Σ=0 and 
# E = -ϕ
# so this formula is valid both at fixed and at optimized x
function all_therm_func(app, op::OrderParams, ep::ExtParams)
    @extract op: m
    ϕ = free_entropy(app,op, ep)
    E = -ϕ - m*im_fun(app, op, ep, m)
    Σ = m*(ϕ + E)
    return ThermFunc(ϕ, Σ, E)
end

#################  SADDLE POINT  ##################
# fqh0(app, op, ep) = -2/op.m * ep.α * ∂q0_Ge(app, op, ep)
fqh0(app, op, ep) = fqh0!(app, op, ep)
fqh1(app, op, ep) = 2ep.α * ∂δq_Ge(app, op, ep)
fρh(app, op, ep) = ep.α * ∂ρ_Ge(app, op, ep)

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

function im_fun(app, op::OrderParams, ep::ExtParams, m)
    @extract op: q0 δq qh0 qh1 δqh ρh
    @extract ep: α ρ
    op.m = m # for Ge
    ∂m_Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) + ∂m_Gs(qh0,qh1,δqh,ρh,m) + α*∂m_Ge(app, op, ep)
end

function im(app, op::OrderParams, ep::ExtParams, m₀, atol=1e-4)
    ok, m, it, normf0 = findroot(m -> im_fun(app, op, ep, m), m₀, NewtonMethod(atol=atol))
    ok || error("im failed: m=$m, it=$it, normf0=$normf0")
    return ok, m
end

###############################


function converge!(app::TheApp, op::OrderParams, ep::ExtParams; fixm = false, fixρ = true)
    @extract app : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    it = 0
    reset_timer!()
    for it = 1:maxiters
        Δ = 0.0
        verb > 1 && println("it=$it")

        @update  op.qh0    fqh0       Δ ψ verb  app op ep
        @update  op.qh1    fqh1       Δ ψ verb  app op ep
        @updateI op.δqh ok   iδqh     Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        if fixρ
            @updateI op.ρh ok   iρh   Δ ψ verb  ep.ρ op.qh0 op.qh1 op.δqh op.m
        else
            @update  op.ρh  fρh       Δ ψ verb  app op ep
        end

        # fix_inequalities_hat!(op, ep)
        # fix_inequalities_nonhat!(op, ep)        

        @update op.q0   fq0       Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        @update op.δq   fδq       Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        if !fixρ
            @update ep.ρ   fρ     Δ ψ verb  op.qh0 op.qh1 op.δqh op.ρh op.m
        end

        if !fixm 
            @updateI op.m ok   im    Δ ψ verb  app op ep op.m
        end

        verb > 1 && println(" Δ=$Δ\n")
        verb > 2 && println(all_therm_func(app, op, ep))

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end
    verb > 1 && (print_timer();println())

    ok
end

function converge(;
        q0=0.1, δq=0.5,
        qh0=0., qh1=0., δqh=0.6,
        m = 1.00001, ρ=0, ρh=0,
        α=0.1,
        ϵ=1e-6, maxiters=100000, verb=2, ψ=0.,
        fixm = false, fixρ=true,
        RES=600, limit=15, width=6 
    )
    op = OrderParams(q0,δq,qh0,qh1,δqh, ρh,m)
    ep = ExtParams(α, ρ)
    app = TheApp(RES, limit, width, ϵ, ψ, maxiters, verb)
    converge!(app, op, ep, fixm=fixm, fixρ=fixρ)
    tf = all_therm_func(app, op, ep)
    println(tf)
    return app, op, ep, tf
end


function span(;
        q0=0.1, δq=0.5,
        qh0=0., qh1=0., δqh=0.6, ρh=0,
        m=0.1, ρ=0.384312, α=1,
        ϵ=1e-6, maxiters=10000,verb=2, ψ=0.,
        RES=600, limit=15, width=6, 
        kws...)

    op = OrderParams(q0,δq,qh0,qh1,δqh,ρh, first(m))
    ep = ExtParams(first(α), first(ρ))
    app = TheApp(RES, limit, width, ϵ, ψ, maxiters, verb)
    return span!(app, op, ep; ρ=ρ,α=α,m=m, kws...)
end

function span!(app::TheApp, op::OrderParams, ep::ExtParams;
        m=0,  α=1, ρ=1,
        resfile = "results.txt",
        fixm=false, fixρ=false)

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end

    results = []
    for m in m, α in α, ρ in ρ
        fixm && (op.m = m)
        fixρ && (ep.ρ = ρ)
        ep.α = α;
        println("# NEW ITER: α=$(ep.α)  ρ=$(ep.ρ)  m=$(op.m)")

        if fixm
            ok = converge!(app, op, ep, fixm=true, fixρ=fixρ)
        else
            ok = findSigma0!(app, op, ep; tol=app.ϵ)
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
        tf = all_therm_func(app, op, ep)
        tf.Σ < -1e-7 && @warn("Sigma negative")
        push!(results, (ok, deepcopy(ep), deepcopy(tf), deepcopy(op)))
        ok && open(resfile, "a") do rf
            println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
        end
        !ok && break
        app.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end


function findSigma0!(app, op, ep;
                tol = 1e-4, dm = 10, smallsteps = true)
    mlist = Any[]
    Σlist = Any[]

    ###PRIMO TENTATIVO
    println("@@@ T 1 : m=$(op.m)")
    ok = converge!(app, op, ep, fixm=true, fixρ=true)
    tf = all_therm_func(app, op, ep)
    println(tf)
    push!(mlist, op.m)
    push!(Σlist, tf.Σ)
    absSigma = abs(tf.Σ)

    println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")
    ###SECOND TENTATIVO
    if absSigma > tol
        op.m += abs(op.m * tf.Σ * dm) > 0.5 ? 0.5*sign(op.m * tf.Σ * dm) : op.m * tf.Σ * dm
        println("@@@ T 2 : m=$(op.m)")

        ok = converge!(app, op, ep, fixm=true, fixρ=true)
        tf = all_therm_func(app, op, ep)
        println(tf)
        push!(mlist, op.m)
        push!(Σlist, tf.Σ)
        absSigma = abs(tf.Σ)
        println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")
    end

    ###ALTRI  TENTATIVI
    trial = 3
    while absSigma > tol
        s = 0
        if trial >= 3
            s = -(mlist[end]*Σlist[end-1] - mlist[end-1]*Σlist[end])/(Σlist[end]-Σlist[end-1])
        end
        if smallsteps && abs(s - op.m) >  op.m * abs(tf.Σ) * dm
            op.m += sign(s - op.m) * min(op.m * abs(tf.Σ) * dm, 0.5)
        else
            op.m = s
        end
        println("@@@ T $(trial) : m=$(op.m)")
        ok = converge!(app, op, ep, fixm=true, fixρ=true)

        tf = all_therm_func(app, op, ep)
        println(tf)
        println("\n@@@ m=$(op.m) Σ=$(tf.Σ) \n")
        push!(mlist, op.m)
        push!(Σlist, tf.Σ)
        absSigma = abs(tf.Σ)
        trial += 1
    end

    return ok
end

function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    ep = ExtParams(res[line,1:2]...)
    op = OrderParams(res[line,6:end]...)
    tf = ThermFunc(res[line,3:5]...)
    return ep, op, tf
end

end ## module
