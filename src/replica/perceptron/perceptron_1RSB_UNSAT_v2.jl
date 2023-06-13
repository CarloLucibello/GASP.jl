module Perc

using LittleScienceTools.Roots
using OffsetArrays
using TimerOutputs

include("../../common.jl")

ENABLE_TIMINGS = true

macro t(exprs...)
    if ENABLE_TIMINGS
        return :(@timeit($(esc.(exprs)...)))
    else
        return esc(exprs[end])
    end
end

###### INTEGRATION  ######

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
        Ge = 0.
        new(RES, LIMIT, WIDTH, 1/RES,        
            kernel, GEs, Ge,
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
Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) = (δqh - 2*ρ*ρh - δq*qh1 + q0*qh0*m - qh1*m)/2

∂m_Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) = (q0*qh0 - qh1) / 2

#### ENTROPIC TERM ####

Gs(qh0,qh1,δqh,ρh,m) = 0.5*((Power(ρh,2) + qh0)/(δqh + (qh0 - qh1)*m) + Log(δqh)/m - Log(δqh + (qh0 - qh1)*m)/m)

∂m_Gs(qh0,qh1,δqh,ρh,m) =  (-((qh0 - qh1)/(m*(δqh + m*(qh0 - qh1)))) - 
    ((qh0 - qh1)*(qh0 + Power(ρh,2)))/Power(δqh + m*(qh0 - qh1),2) - 
    Log(δqh)/Power(m,2) + Log(δqh + m*(qh0 - qh1))/Power(m,2))/2

#### ENERGETIC TERM ####

@timeit function firststepGe!(gK, theapp, δq)
    @extract theapp: a LIMIT WIDTH
    
    fK(h) = begin
            c = h / √δq
            c > 0 && return 0.
            -1/2 * min(c^2, 4.) 
        end
    
    for i=-LIMIT:LIMIT
        h = a * i
        gK[i] = fK(h)
    end
    extend_linear!(gK)
end

function init_kernel!(theapp::TheApp, dQ)
    @extract theapp: a kernel WIDTH
    if dQ > a^2/100.     #protection for small deltaQ
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
    @extract theapp: kernel LIMIT WIDTH
    expg = symvec(LIMIT+WIDTH)
    for i=-(LIMIT+WIDTH):LIMIT+WIDTH
        expg[i] = exp(m*g[i])
    end
    init_kernel!(theapp, dQ)
   
    # need a function barrier for performace (x10)
    # Why? investigate type instabilities
    onestepGe_inner!(gnew, expg, kernel, LIMIT, WIDTH, m)
    # @inbounds for i=-LIMIT:LIMIT
    #     # s = AccumulaTrick()
    #     s = 0. 
    #     @unsafe for k=-WIDTH:WIDTH
    #         s += expg[i+k]*kernel[k]
    #     end
    #     gnew[i] = log(value(s)) / m
    # end
    extend_linear!(gnew)
end

@timeit function laststepGe(theapp::TheApp, g, q0, ρ)
    @extract theapp: kernel LIMIT WIDTH a
    init_kernel!(theapp, q0)
    c = -ρ * √(1/(q0*(q0-ρ^2))) * a
    s = AccumulaTrick()
    for k=-WIDTH:WIDTH
        s += g[k] * kernel[k] * 2H(c*k)
    end
    value(s)
end

function computeGe!(theapp::TheApp, op::OrderParams, ep::ExtParams)
    @extract theapp: GEs
    @extract op: m q0 δq 
    @extract ep: ρ

    firststepGe!(GEs[1], theapp, δq)
    onestepGe!(GEs[0], theapp, GEs[1], m, 1 - q0)
    theapp.Ge = laststepGe(theapp, GEs[0], q0, ρ)
    theapp.Ge
end

function ∂q0_Ge(theapp::TheApp, op::OrderParams, ep::ExtParams)
    δ = 1e-6
    op.q0 += δ
    f1 = computeGe!(theapp, op, ep)
    op.q0 -= δ
    f0 = computeGe!(theapp, op, ep)
    (f1-f0) / δ
end

function ∂δq_Ge(theapp::TheApp, op::OrderParams, ep::ExtParams)
    δ = 1e-6
    op.δq += δ
    f1 = computeGe!(theapp, op, ep)
    op.δq -= δ
    f0 = computeGe!(theapp, op, ep)
    (f1-f0) / δ
end

function ∂m_Ge(theapp::TheApp, op::OrderParams, ep::ExtParams)
    δ = 1e-6
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
fqh0(app, op, ep) = -2/op.m * ep.α * ∂q0_Ge(app, op, ep)
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

function im(app, op::OrderParams, ep::ExtParams, m₀, atol=1e-8)
    ok, m, it, normf0 = findroot(m -> im_fun(app, op, ep, m), m₀, NewtonMethod(atol=atol))
    ok || error("im failed: m=$m, it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
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

        if !fixm ## && it%5==0
            @updateI op.m ok   im    Δ ψ verb  app op ep op.m
        end

        verb > 1 && println(" Δ=$Δ\n")
        verb > 1 && println(all_therm_func(app, op, ep))

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
    end
    print_timer();println()

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
        println("OK $ok")
        !ok && break
        app.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
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
