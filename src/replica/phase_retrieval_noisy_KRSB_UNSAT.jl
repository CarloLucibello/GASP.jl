module PhaseRetr

using LittleScienceTools.Roots
using OffsetArrays
using TimerOutputs
using QuadGK
import LsqFit: curve_fit
import IterTools: product
using Cubature
# using Plots; plotlyjs(size=(1000,800))
include("../common.jl")
include("interpolation.jl")

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
        r = G(z) .* f(z)
        # isfinite(r) ? r : 0.0
    end, int..., abstol=1e-7, maxevals=10^7)[1]


## Cubature.jl

∫∫D(f, xmin::Vector, xmax::Vector) = hcubature(z->begin
            G(z[1])*G(z[2])*f(z[1],z[2])
            # isfinite(r) ? r : 0.0
        end, xmin, xmax, abstol=1e-7)[1]


∫∫D(fdim, f, xmin::Vector, xmax::Vector) = hcubature(fdim, (z,y)->begin
        y .= (G(z[1]).*G(z[2])).*f(z[1],z[2])
        # isfinite(r) ? r : 0.0
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
        a[k] = deepcopy(x0)
    end
    a
end

function qvec(a::Vector{T}) where T
    K = length(a) - 2
    q = QVec{T}(0:K+1)
    for k=0:K+1
        q[k] = a[k+1]
    end
    q
end

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
extend_linear!(v::ExtVec) = (extend_left!(v); extend_right!(v))
extend_zero!(v::ExtVec) = (v.a_left=v.b_left=v.a_right=v.b_right=0.)

############### PARAMS ################

mutable struct OrderParams
    q::QVec{Float64}
    δq::Float64
    qh::QVec{Float64}
    δqh::Float64
    m::QVec{Float64}
    ρh::Float64
end

function extrapolate!(op, ops::Vector{OrderParams})
    ord = length(ops) - 2 #fit order
    model(x, p) = sum(p[i+1]./ x.^i for i=0:ord)
    # model(x, p) = sum(p[i+1]./ x.^i for i=0:ord-2) + p[ord]/x.^p[ord+1]
    for k in linearindices(op.q)
        p₀ = [op.q[k]; zeros(ord)]
        op.q[k] = curve_fit(model, 1:length(ops), [o.q[k] for o in ops], p₀).param[1]
    end
    for k in linearindices(op.qh)
        p₀ = [op.qh[k]; zeros(ord)]
        op.qh[k] = curve_fit(model, 1:length(ops), [o.qh[k] for o in ops], p₀).param[1]
    end

    p₀ = [op.δqh; zeros(ord)]
    op.δqh = curve_fit(model, 1:length(ops), [o.δqh for o in ops], p₀).param[1]
    p₀ = [op.δq; zeros(ord)]
    op.δq = curve_fit(model, 1:length(ops), [o.δq for o in ops], p₀).param[1]
    p₀ = [op.ρh; zeros(ord)]
    op.ρh = curve_fit(model, 1:length(ops), [o.ρh for o in ops], p₀).param[1]
    op
end

mutable struct ExtParams
    α::Float64
    ρ::Float64
    Δ::Float64
end

mutable struct TheApp
    K::Int # number of replica symmetry breakings (K-RSB)
    RES::Int # 1/a
    LIMIT::Int # width of Ge
    WIDTH::Int # kernel size
    a::Float64 # lattice spacing

    kernel::SymVec
    GEs::QVec{ExtVec} # Ge[k] = Ge at level k of the ultrametric tree. k=0,..,K
    pH::QVec{ExtVec}  # cavity fields

    λ::QVec{Float64}  # Eigenvalues of Q matrix
    λh::QVec{Float64} # Eigenvalues of Qh matrix

    Gi::Float64 
    Gs::Float64 
    Ge::Float64 

    ϵ::Float64 # stop criterium
    ψ::Float64 # dumping
    maxiters::Int
    verb::Int

    function TheApp(K, RES, limit, width, ϵ, ψ, maxiters, verb)
        LIMIT = limit*RES
        WIDTH = width*RES
        kernel = symvec(WIDTH)
        GEs = qvec(0:K, ExtVec(LIMIT))
        pH = qvec(0:K, ExtVec(LIMIT))
        Ge = 0.
        λ = qvec(0:K+1, 0.)
        λh = qvec(0:K+1, 0.)
        new(K, RES, LIMIT, WIDTH, 1/RES,       
            kernel, GEs, pH,
            λ, λh,
            0, 0, 0,
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
function  computeGi!(app, op, ep)
    @extract app: K  
    @extract op: q δq qh δqh ρh m 
    @extract ep: ρ

    Gi = 0.
    for k=0:K-1
        Gi += q[k] * qh[k] * (m[k+1] - m[k])
    end
    Gi += - m[K]*q[K]*qh[K] - δq*qh[K] + δqh*q[K]
    app.Gi =  0.5*Gi - ρ*ρh
end
# Gi(q0,δq,qh0,qh1,δqh,ρh,m,ρ) = (δqh - 2*ρ*ρh - δq*qh1 + q0*qh0*m - qh1*m)/2

∂m_Gi(app, op, ep) = (op.q[0]*op.qh[0] - op.qh[1]*op.q[1]) / 2

#### ENTROPIC TERM ####


function computeGs!(theapp::TheApp, op::OrderParams)
    @extract theapp: λ λh K 
    @extract op: m qh ρh δqh

    λh[K+1] = -δqh
    for k=K:-1:1
        λh[k] = λh[k+1] + m[k]*(qh[k] - qh[k-1])
    end
    for k=1:K+1
        λ[k] = -1/λh[k]
    end

    # Gs = q[0] / λ[1]
    # for k=1:K-1
    #     Gs += (log(λ[k+1])- log(λ[k])) / m[k]
    # end
    # Gs += log(λ[K+1])
    Gs = -(qh[0]+ρh^2) / λh[1]
    for k=1:K-1
        Gs += -(log(-λh[k]) - log(-λh[k+1])) / m[k]
    end
    Gs += -log(-λh[K]) / m[K]  + log(δqh) / m[K]
    @assert isfinite(Gs)
    theapp.Gs = 0.5*Gs
end

function compute_q_δq_δqh(theapp::TheApp, op::OrderParams)
    @extract theapp: λ λh K
    @extract op: qh ρh m
    @assert m[K+1] == 1
    @assert m[0] == 0
    qnew = qvec(0:K+1, 0.)
    δq = 0.
    f(δqh) = begin
        λh[K+1] = -δqh
        for k=K:-1:1
            λh[k] = λh[k+1] + m[k]*(qh[k] - qh[k-1])
        end
 
        for k=1:K+1
            λ[k] = -1/λh[k]
        end
        # qnew[0] = qh[0] * λ[1]^2 - ρh^2 # check
        qnew[0] = (qh[0] + ρh^2)* λ[1]^2
        for k=1:K-1
            qnew[k] = qnew[k-1] + (λ[k] - λ[k+1]) / m[k]
        end
        qnew[K] = qnew[K+1] = 1
 
        δq = m[K]*(qnew[K-1]-qnew[K]) + λ[K]
        return 1 / δq
    end
    

    δqh₀ = op.δqh
    ok, δqh, it, normf0 = findroot(δqh -> f(δqh) - δqh, δqh₀, NewtonMethod(atol=1e-8,verb=0))
    ok || error("iδqh failed: iδqh=$(δqh), it=$it, normf0=$normf0")
    return ok, qnew, δq, δqh
end

function ∂m_Gs(theapp::TheApp, op::OrderParams)
    δ = 1e-5
    op.m[end-1] += δ
    f1 = computeGs!(theapp, op)
    op.m[end-1] -= δ
    f0 = computeGs!(theapp, op)
    (f1-f0) / δ
end

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

@timeit function firststepGe!(gK, theapp, δq, ρ, t, ξ)
    @extract theapp: a LIMIT WIDTH
    
    fK(h) = -argGe(t^2 + ξ, δq, ρ*t + h)
    g = gK.v #symmetric vector
    @unsafe for i=-LIMIT:LIMIT
        h = a * i
        g[i] = fK(h)
        @assert isfinite(g[i]) "t=$t i=$i h=$h $(g[i])"
    end
    extend_linear!(gK)
end

function init_kernel!(kernel, a, WIDTH, dQ)
    if dQ > a^2/100     #protection for small deltaQ
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
            # @assert isfinite(s) "here $(expg[i+k]) $(kernel[k])"
        end
        gnew[i] = log(value(s)) / m
        @assert isfinite(gnew[i])
    end
end

@timeit function onestepGe!(gnew, theapp::TheApp, g, m, dQ)
    @extract theapp: kernel LIMIT WIDTH a
    expg = symvec(LIMIT+WIDTH)
    for i=-(LIMIT+WIDTH):(LIMIT+WIDTH)
        expg[i] = exp(m*g[i])
        !isfinite(expg[i]) && (expg[i] = 0) # protection from overflow
        # @assert isfinite(expg[i]) "$(g[i])"
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
    s = AccumulaTrick()
    for k=-WIDTH:WIDTH
        s += g[k] * kernel[k]
    end
    value(s)
end

function computeGe!(theapp::TheApp, op::OrderParams, ep::ExtParams)
    @extract theapp: GEs WIDTH a K
    @extract op: m q δq 
    @extract ep: ρ Δ

    theapp.Ge = ∫∫D((t,ξ) -> begin
            firststepGe!(GEs[K], theapp, δq, ρ, t, √Δ*ξ)
            for k=K:-1:1
                onestepGe!(GEs[k-1], theapp, GEs[k], m[k], q[k] - q[k-1])
            end
            laststepGe(theapp, GEs[0], q[0] - ρ^2)
        end)
    
    theapp.Ge
end

M2G(g::ExtVec, i::Int, a::Float64) = ((g[i+1] - g[i-1])/(2.*a))^2

function computeQh_inner!(P, G, scra, kernel, LIMIT, WIDTH, a, m)
    ss = AccumulaTrick()
    @inbounds for i=-LIMIT:LIMIT
        # s = AccumulaTrick()
        s = 0.
        @unsafe for j=-WIDTH:WIDTH
            s += scra[i-j]*kernel[j]
        end
        P[i] = value(s) * exp(+m * G[i])
        !isfinite(P[i]) && (P[i]=0)
        ss += M2G(G, i, a) * P[i]
    end
    value(ss)
end

function computeQh!(theapp::TheApp, op::OrderParams, ep::ExtParams)
    @extract theapp: a kernel K  LIMIT WIDTH GEs pH
    @extract op: m q δq 
    @extract ep: ρ α Δ

    scra = symvec(LIMIT+WIDTH)
   
    # STEP k=0
    
    res = ∫∫D((t,ξ) -> begin
            qht = zeros(K+1)
            firststepGe!(GEs[K], theapp, δq, ρ, t, √Δ*ξ)
            for k=K:-1:1
                onestepGe!(GEs[k-1], theapp, GEs[k], m[k], q[k] - q[k-1])
            end        
            laststepGe(theapp, GEs[0], q[0] - ρ^2)
            # K = 0
            init_kernel!(kernel, a, WIDTH, q[0] - ρ^2)
            s = 0.
            P = pH[0]
            G = GEs[0]
            for i=-WIDTH:WIDTH
                P[i] = kernel[i]
                s += M2G(G, i, a) * P[i]
                # @assert isfinite(G[i])
                # @assert isfinite(s)
            end
            for i=WIDTH+1:LIMIT
                P[i] = P[-i] = 0
            end
            qht[1] = value(s)
            extend_zero!(P)
            @assert isfinite(qht[1])
            # K > 0
            for k=1:K    
                P = pH[k-1]
                G = GEs[k-1]
                for i=-(LIMIT+WIDTH):(LIMIT+WIDTH)
                    scra[i] = P[i] * exp(-m[k] * G[i])
                    # !isfinite(scra[i]) && (scra[i] = 0)
                    # @assert isfinite(scra[i]) "isfinite(scra[i]) k=$k i=$i $(scra[i]) g[i]=$(g[i]) p[i]=$(p[i])"
                end
                # @assert isapprox(sum(i->P[i], -LIMIT:LIMIT), 1, atol=1e-7)

                P = pH[k]
                G = GEs[k]
                init_kernel!(kernel, a, WIDTH, q[k] - q[k-1])
                qht[k+1] = computeQh_inner!(P, G, scra, kernel, LIMIT, WIDTH, a, m[k])
                extend_zero!(P)
                @assert isfinite(qht[k+1])
                # @assert isapprox(sum(i->P[i], -LIMIT:LIMIT), 1, atol=1e-7)
            end
            qht
        end)

    
    qh = qvec(0:K+1, 0.)
    for k =0:K
        qh[k] = res[k+1]*α
    end
    qh[K+1] = qh[K]

    return qh
end

function ∂q0_Ge(theapp::TheApp, op::OrderParams, ep::ExtParams)
    δ = 1e-5
    op.q[0] += δ
    f1 = computeGe!(theapp, op, ep)
    op.q[0] -= δ
    f0 = computeGe!(theapp, op, ep)
    (f1-f0) / δ
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
    op.m[end-1] += δ
    f1 = computeGe!(theapp, op, ep)
    op.m[end-1] -= δ
    f0 = computeGe!(theapp, op, ep)
    (f1-f0) / δ
end

############ Thermodynamic functions

function free_entropy(app, op::OrderParams, ep::ExtParams)
    @extract ep: α

    Ge = computeGe!(app, op, ep)
    Gs = computeGs!(app, op)
    Gi = computeGi!(app, op, ep)
    
    Gi + Gs + α*Ge
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
    E = -ϕ - m[end-1]*im_fun(app, op, ep, m[end-1])
    Σ = m[end-1]*(ϕ + E)
    return ThermFunc(ϕ, Σ, E)
end

#################  SADDLE POINT  ##################
fqh0(app, op, ep) = -2/op.m[1] * ep.α * ∂q0_Ge(app, op, ep)
fqh1(app, op, ep) = 2ep.α * ∂δq_Ge(app, op, ep)


fρh(app, op, ep) = ep.α * ∂ρ_Ge(app, op, ep)

fρ(ρh, λ1) = -ρh / λ1 #check SIGN

iρh(ρ, λ1) = (true, -ρ*λ1) #check SIGN


function im_fun(app, op::OrderParams, ep::ExtParams, m)
    @extract ep: α
    op.m[end-1] = m # for Ge
    ∂m_Gi(app, op, ep) + ∂m_Gs(app, op) + α*∂m_Ge(app, op, ep)
end

function im(app, op::OrderParams, ep::ExtParams, m₀, atol=1e-4)
    ok, m, it, normf0 = findroot(m -> im_fun(app, op, ep, m), m₀, NewtonMethod(atol=atol))
    ok || error("im failed: m=$m, it=$it, normf0=$normf0")
    return ok, m
end

###############################


function converge!(app::TheApp, op::OrderParams, ep::ExtParams; 
            fixm=false, fixρ=true, extrap=-1)
    @extract app : maxiters verb ϵ ψ K

    Δ = Inf
    ok = false

    it = 0
    ops = Vector{OrderParams}() # keep some history and extrapolate for quicker convergence
    reset_timer!()
    for it = 1:maxiters
        Δ = 0.0
        verb > 1 && println("it=$it")
        tic()

        qh = computeQh!(app, op, ep)
        @update op.qh   identity  Δ ψ verb qh
        # @update  op.qh[0]    fqh0       Δ ψ verb  app op ep
        # @update  op.qh[1]    fqh1       Δ ψ verb  app op ep
        
        if !fixρ
            @update  op.ρh  fρh       Δ ψ verb  app op ep
        end 

        # fix_inequalities_hat!(op, ep)
        # fix_inequalities_nonhat!(op, ep)        
        
        ok, q, δq, δqh = compute_q_δq_δqh(app, op)
        @update op.δqh   identity       Δ ψ verb δqh
        @update op.q     identity       Δ ψ verb q
        @update op.δq    identity       Δ ψ verb δq
        
        if fixρ
            @updateI op.ρh ok   iρh   Δ ψ verb  ep.ρ app.λh[1]
        else    
            @update ep.ρ   fρ     Δ ψ verb  op.ρh app.λh[1]
        end

        # if !fixm 
        #     @updateI op.m ok   im    Δ ψ verb  app op ep op.m
        # end

        verb > 1 && println(" Δ=$Δ\n")
        verb > 2 && it%5==0 && (println(op);println(all_therm_func(app, op, ep)))

        @assert isfinite(Δ)
        ok = ok && Δ < ϵ
        ok && break
        toc()
        extrap > 0 && it > extrap && push!(ops, deepcopy(op))
        if extrap > 0 && it > extrap && it % extrap == 0
            extrapolate!(op, ops)
            empty!(ops)
            verb > 1 && println("# estrapolation -> $op \n")
        end
    end
    verb > 1 && (print_timer();println())

    ok
end

function interpq(q, K)
    Kold = length(q) - 2
    fq = Interp(i->q[i],1:Kold+1)
    qnew = [fq(1 + i*Kold/K) for i=0:K]
    for i=2:length(qnew)-1
        qnew[i] < qnew[i-1] && (qnew[i] = qnew[i-1]+1e-5)
    end
    [qnew; q[end]]
end

function converge(;
        K = 1,
        q0=0.584629, δq=2.24397, qh0=0.00236847, qh1=0.017943, δqh=0.445676, ρh=0.0420651, 
        qh = nothing, q = nothing,
        m=23.2139, 
        α=1.4, ρ = 0.5, Δ =0.,
        ϵ=1e-4, maxiters=1000, verb=2, ψ=0.,
        fixm=true, fixρ=true, extrap=5,
        RES=100, limit=12, width=6 
    )
    
    qs = q != nothing ? qvec(interpq(q,K)) : qvec([linspace(q0,1., K+1); 1.])
    qhs = qh != nothing ? qvec(interpq(qh,K)) : qvec([linspace(qh0,qh1,K+1); qh1])
    
    ms = qvec([sqrt.(linspace(0., 1, K+1)) .* m; 1.])
    op = OrderParams(qs, δq, qhs, δqh, ms, ρh)

    @show ms qs qhs
    
    ep = ExtParams(α, ρ, Δ)
    app = TheApp(K, RES, limit, width, ϵ, ψ, maxiters, verb)
    converge!(app, op, ep, fixm=fixm, fixρ=fixρ, extrap=extrap)
    tf = all_therm_func(app, op, ep)
    println(tf)
    return app, op, ep, tf
end


function span(;
            K = 1,
            q0=0.584629, δq=2.24397, qh0=0.00236847, qh1=0.017943, δqh=0.445676, ρh=0.0420651, 
            qh = nothing, q = nothing,
            m=23.2139, 
            α=1.4, ρ = 0.5, Δ =0.,
            ϵ=1e-4, maxiters=1000, verb=2, ψ=0.,
            RES=100, limit=12, width=6, 
            kws...)

    qs = q != nothing ? qvec(interpq(q,K)) : qvec([linspace(q0,1., K+1); 1.])
    qhs = qh != nothing ? qvec(interpq(qh,K)) : qvec([linspace(qh0,qh1,K+1); qh1])

    for k=1:K÷2
        qs[k] = qs[0] + k*1e-2
        qhs[k] = qhs[0] + k*1e-2
    end

    for k=1:K÷2
        qs[K-k] = qs[K] - k*1e-2
        qhs[K-k] = qs[K] - k*1e-2
    end

    ms = qvec([sqrt.(linspace(0., 1, K+1)) .* m; 1.])
    op = OrderParams(qs, δq, qhs, δqh, ms, ρh)

    @show ms qs qhs

    ep = ExtParams(first(α), first(ρ), Δ)
    app = TheApp(K, RES, limit, width, ϵ, ψ, maxiters, verb)

    return span!(app, op, ep; ρ=ρ,α=α,kws...)
end

function span!(app::TheApp, op::OrderParams, ep::ExtParams;
        α=1, ρ=1,
        resfile = "results.txt",
        extrap=5, # if positive extrapolate op every extrap steps
        fixm=false, fixρ=false)

    !isfile(resfile) && open(resfile, "w") do f
        allheadersshow(f, ExtParams, ThermFunc, OrderParams)
    end


    results = []
    for α in α, ρ in ρ
        fixρ && (ep.ρ = ρ)
        ep.α = α;
        println("# NEW ITER: α=$(ep.α)  ρ=$(ep.ρ)")

        if fixm
            ok = converge!(app, op, ep, fixm=true, fixρ=fixρ, extrap=extrap)
        else
            ok = findSigma0!(app, op, ep; tol=app.ϵ, fixρ=fixρ, extrap=extrap)
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
                fixρ=true, extrap=-1,
                tol = 1e-4, dm = 10, smallsteps=true)
    mlist = Any[]
    Σlist = Any[]

    ###PRIMO TENTATIVO
    println("@@@ T 1 : m=$(op.m)")
    ok = converge!(app, op, ep, fixm=true, fixρ=fixρ, extrap=extrap)
    tf = all_therm_func(app, op, ep)
    println(tf)
    push!(mlist, op.m[end-1])
    push!(Σlist, tf.Σ)
    absSigma = abs(tf.Σ)

    println("\n@@@ m=$(op.m[end-1]) Σ=$(tf.Σ) \n")
    ###SECOND TENTATIVO
    if absSigma > tol
        op.m[end-1] += abs(op.m[end-1] * tf.Σ * dm) > 0.5 ? 0.5*sign(op.m[end-1] * tf.Σ * dm) : op.m[end-1] * tf.Σ * dm
        println("@@@ T 2 : m=$(op.m)")

        ok = converge!(app, op, ep, fixm=true, fixρ=fixρ)
        tf = all_therm_func(app, op, ep)
        println(tf)
        push!(mlist, op.m[end-1])
        push!(Σlist, tf.Σ)
        absSigma = abs(tf.Σ)
        println("\n@@@ m=$(op.m[end-1]) Σ=$(tf.Σ) \n")
    end

    ###ALTRI  TENTATIVI
    trial = 3
    while absSigma > tol
        s = 0
        if trial >= 3
            s = -(mlist[end]*Σlist[end-1] - mlist[end-1]*Σlist[end])/(Σlist[end]-Σlist[end-1])
        end
        if smallsteps && abs(s - op.m[end-1]) >  op.m[end-1] * abs(tf.Σ) * dm
            op.m[end-1] += sign(s - op.m[end-1]) * min(op.m[end-1] * abs(tf.Σ) * dm, 0.5)
        else
            op.m[end-1] = s
        end
        println("@@@ T $(trial) : m=$(op.m[end-1])")
        ok = converge!(app, op, ep, fixm=true, fixρ=fixρ)

        tf = all_therm_func(app, op, ep)
        println(tf)
        println("\n@@@ m=$(op.m[end-1]) Σ=$(tf.Σ) \n")
        push!(mlist, op.m[end-1])
        push!(Σlist, tf.Σ)
        absSigma = abs(tf.Σ)
        trial += 1
    end

    return ok
end

function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line # -1 since readdlm discards the header
    @show res[line,:]
    # ep = ExtParams(res[line,1:2]...)
    # op = OrderParams(res[line,6:end]...)
    # tf = ThermFunc(res[line,3:5]...)
    # return ep, op, tf
end

end ## module
