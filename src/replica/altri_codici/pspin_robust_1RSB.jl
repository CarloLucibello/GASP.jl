using ExtractMacro
using LittleScienceTools.Roots

mutable struct OrderParams
    q0::Float64
    q1::Float64
    qb0::Float64
    qb1::Float64
    qb2::Float64
    qh0::Float64
    qh1::Float64
    qh2::Float64
    qbh0::Float64
    qbh1::Float64
    qbh2::Float64
    m::Float64
    mh::Float64
    x::Float64
    y::Float64
end

mutable struct ExtParams
    β::Float64
    p::Int
    r::Float64 #field strength
    k::Int  #field exponent
    g::Float64 # γ^2 /λt
end


mutable struct Params
    ϵ::Float64
    ψ::Float64  #dumping
    maxiters::Int
    verb::Int
end

mutable struct ThermFunc
    ϕ::Float64 #free entropy
    Σ::Float64
    f::Float64 # replicated system's pure state free energy (is of order O(y))
end

function all_therm_func(op::OrderParams, ep::ExtParams)
    ϕ = free_entropy(op, ep)
    f = free_ene_state(op, ep)
    Σ = (ϕ + ep.β *f)*op.x

    return ThermFunc(ϕ, Σ, f)
end

# CForm mathematica
Power(x, a) = x^a
Log(x) = log(x)

## There is a shift of 0.5*y with respect to the usual definition
function free_entropy(op::OrderParams, ep::ExtParams)
    @extract ep: β p r k g
    @extract op: q0 q1 qb0 qb1 qb2 qh0 qh1 qh2 qbh0 qbh1 qbh2 y m mh x

    ϕ = (((-qbh0 + qh0)*(-1 + y))/
      (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x + qh0*x) -
     2*m*mh*y + (qh0 + qbh0*(-1 + y) + Power(mh,2)*y)/
      (-qh2 - qh1*(-1 + x) + qh0*x -
        (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)) +
     ((1 - x*(Power(q0,p) + Power(qb0,p)*(-1 + y)) +
          (-1 + x)*(Power(q1,p) + Power(qb1,p)*(-1 + y)) +
          Power(qb2,p)*(-1 + y))*y*Power(β,2))/2. -
     y*(qh2 - x*(q0*qh0 + qb0*qbh0*(-1 + y)) +
        (-1 + x)*(q1*qh1 + qb1*qbh1*(-1 + y)) + qb2*qbh2*(-1 + y) -
        g*(1 + qb2*(-1 + y))*β) + 2*Power(m,k)*r*y*β/k  -
     (1 - 1/x)*((-1 + y)*Log(-qbh1 + qbh2 + qh1 - qh2) +
        Log(qh1 - qh2 - (-qbh1 + qbh2)*(-1 + y))) -
     ((-1 + y)*Log(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) -
           qbh0*x + qh0*x) +
        Log(-qh2 - qh1*(-1 + x) + qh0*x -
          (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)))/x)

    return ϕ / 2
end

# replicated system pure state'free energy
# f = -∂(x*ϕ)/∂x / β 
function free_ene_state(op::OrderParams, ep::ExtParams)
    @extract ep: β p r k g
    @extract op: q0 q1 qb0 qb1 qb2 qh0 qh1 qh2 qbh0 qbh1 qbh2 y m mh x

    dϕ = ((-(((qh0 - qh1 - (-qbh0 + qbh1)*(-1 + y))/(-qh2 - qh1*(-1 + x) + qh0*x - (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)) + 
    ((-qbh0 + qbh1 + qh0 - qh1)*(-1 + y))/(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x + qh0*x))/x) - 
((-qbh0 + qh0)*(-qbh0 + qbh1 + qh0 - qh1)*(-1 + y))/Power(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x + qh0*x,2) - (-(q0*qh0) + q1*qh1 - qb0*qbh0*(-1 + y) + qb1*qbh1*(-1 + y))*y - 
((qh0 - qh1 - (-qbh0 + qbh1)*(-1 + y))*(qh0 + qbh0*(-1 + y) + Power(mh,2)*y))/Power(-qh2 - qh1*(-1 + x) + qh0*x - (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y),2) + 
((-Power(q0,p) + Power(q1,p) - Power(qb0,p)*(-1 + y) + Power(qb1,p)*(-1 + y))*y*Power(β,2))/2. - 
((-1 + y)*Log(-qbh1 + qbh2 + qh1 - qh2) + Log(qh1 - qh2 - (-qbh1 + qbh2)*(-1 + y)))/Power(x,2) + 
((-1 + y)*Log(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x + qh0*x) + Log(-qh2 - qh1*(-1 + x) + qh0*x - (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)))/Power(x,2))/2.)

    return -(x*dϕ + free_entropy(op,ep))/β
end

function complexity(op::OrderParams, ep::ExtParams)
    @extract ep: β p r k g
    @extract op: q0 q1 qb0 qb1 qb2 qh0 qh1 qh2 qbh0 qbh1 qbh2 y m mh x

    return -x^2*(-(((qh0 - qh1 - (-qbh0 + qbh1)*(-1 + y))/
           (-qh2 - qh1*(-1 + x) + qh0*x -
             (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)) +
          ((-qbh0 + qbh1 + qh0 - qh1)*(-1 + y))/
           (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) -
             qbh0*x + qh0*x))/x) -
     ((-qbh0 + qh0)*(-qbh0 + qbh1 + qh0 - qh1)*(-1 + y))/
      Power(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) -
        qbh0*x + qh0*x,2) -
     (-(q0*qh0) + q1*qh1 - qb0*qbh0*(-1 + y) +
        qb1*qbh1*(-1 + y))*y -
     ((qh0 - qh1 - (-qbh0 + qbh1)*(-1 + y))*
        (qh0 + qbh0*(-1 + y) + Power(mh,2)*y))/
      Power(-qh2 - qh1*(-1 + x) + qh0*x -
        (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y),2) +
     ((-Power(q0,p) + Power(q1,p) -
          Power(qb0,p)*(-1 + y) + Power(qb1,p)*(-1 + y))*
        y*Power(β,2))/2. -
     ((-1 + y)*Log(-qbh1 + qbh2 + qh1 - qh2) +
        Log(qh1 - qh2 - (-qbh1 + qbh2)*(-1 + y)))/
      Power(x,2) + ((-1 + y)*
         Log(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) -
           qbh0*x + qh0*x) +
        Log(-qh2 - qh1*(-1 + x) + qh0*x -
          (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)))/
      Power(x,2))/2.
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

function headershow(io::IO, T::Type, i0 = 0)
    print(io, join([string(i+i0,"=",f) for (i,f) in enumerate(fieldnames(T))], " "))
    return i0 + length(fieldnames(T))
end
function allheadersshow(io::IO, x...)
    i0 = 0
    print(io, "#")
    for y in x
        i0 = headershow(io, y, i0)
        print(io, " ")
    end
    println(io)
end

macro update(x, func, Δ, ψ, verb, params...)
    n = string(x.args[2].args[1])
    x = esc(x)
    Δ = esc(Δ)
    ψ = esc(ψ)
    verb = esc(verb)
    params = map(esc, params)
    fcall = Expr(:call, func, params...)
    quote
        oldx = $x
        newx = $fcall
        abserr = abs(newx - oldx)
        relerr = abserr == 0 ? 0 : abserr / ((abs(newx) + abs(oldx)) / 2)
        $Δ = max($Δ, min(abserr, relerr))
        $x = (1 - $ψ) * newx + $ψ * oldx
        $verb > 1 && println("  ", $n, " = ", $x)
    end
end

macro updateI(x, ok, func, Δ, ψ, verb, params...)
    n = string(x.args[2].args[1])
    x = esc(x)
    ok = esc(ok)
    Δ = esc(Δ)
    ψ = esc(ψ)
    verb = esc(verb)
    func = esc(func)
    params = map(esc, params)
    fcall = Expr(:call, func, params...)
    quote
        oldx = $x
        $ok, newx = $fcall
        abserr = abs(newx - oldx)
        relerr = abserr == 0 ? 0 : abserr / ((abs(newx) + abs(oldx)) / 2)
        $Δ = max($Δ, min(abserr, relerr))
        $x = (1 - $ψ) * newx + $ψ * oldx
        $verb > 1 && println("  ", $n, " = ", $x)
    end
end


# fact(p) = factorial(p)
fact(p) = 2

fqh0(β, p, q0) = β^2/fact(p)*p*q0^(p-1)
fqh1(β, p, q1) = β^2/fact(p)*p*q1^(p-1)
fqbh0(β, p, qb0) = β^2/fact(p)*p*qb0^(p-1)
fqbh1(β, p, qb1) = β^2/fact(p)*p*qb1^(p-1)
fqbh2(β, p, g, qb2) = β^2/fact(p)*p*qb2^(p-1) + β*g

fq0(qh0, qh1, qh2, qbh0, qbh1, qbh2, mh, x, y) =  -((1/(-qh2 - qh1*(-1 + x) + qh0*x -
           (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)) -
        (x/(-qh2 - qh1*(-1 + x) + qh0*x -
              (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)) +
           (x*(-1 + y))/
            (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
              qh0*x))/x - ((-qbh0 + qh0)*x*(-1 + y))/
         Power(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
           qh0*x,2) + (-1 + y)/
         (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x + qh0*x)
          - (x*(qh0 + qbh0*(-1 + y) + Power(mh,2)*y))/
         Power(-qh2 - qh1*(-1 + x) + qh0*x -
           (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y),2))/(x*y))

fq1(qh0, qh1, qh2, qbh0, qbh1, qbh2, mh, x, y) = ((-1 + 1/x)*(1/(qh1 - qh2 - (-qbh1 + qbh2)*(-1 + y)) +
         (-1 + y)/(-qbh1 + qbh2 + qh1 - qh2)) -
      ((1 - x)/
          (-qh2 - qh1*(-1 + x) + qh0*x -
            (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)) +
         ((1 - x)*(-1 + y))/
          (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
            qh0*x))/x - ((-qbh0 + qh0)*(1 - x)*(-1 + y))/
       Power(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
         qh0*x,2) - ((1 - x)*(qh0 + qbh0*(-1 + y) + Power(mh,2)*y))/
       Power(-qh2 - qh1*(-1 + x) + qh0*x -
         (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y),2))/((-1 + x)*y)

fq2(qh0, qh1, qh2, qbh0, qbh1, qbh2, mh, x, y) = ((-1 + 1/x)*(-(1/(qh1 - qh2 - (-qbh1 + qbh2)*(-1 + y))) -
         (-1 + y)/(-qbh1 + qbh2 + qh1 - qh2)) -
      (-(1/(-qh2 - qh1*(-1 + x) + qh0*x -
              (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y))) -
         (-1 + y)/
          (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
            qh0*x))/x + ((-qbh0 + qh0)*(-1 + y))/
       Power(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
         qh0*x,2) + (qh0 + qbh0*(-1 + y) + Power(mh,2)*y)/
       Power(-qh2 - qh1*(-1 + x) + qh0*x -
         (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y),2))/y

fqb0(qh0, qh1, qh2, qbh0, qbh1, qbh2, mh, x, y) = -((-((-((x*(-1 + y))/
                (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) -
                  qbh0*x + qh0*x)) + (x*(-1 + y))/
              (-qh2 - qh1*(-1 + x) + qh0*x - (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)))/x) +
        ((-qbh0 + qh0)*x*(-1 + y))/ Power(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
           qh0*x,2) - (-1 + y)/(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x + qh0*x)
          + (-1 + y)/ (-qh2 - qh1*(-1 + x) + qh0*x -
           (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)) -
        (x*(-1 + y)*(qh0 + qbh0*(-1 + y) + Power(mh,2)*y))/
         Power(-qh2 - qh1*(-1 + x) + qh0*x -
           (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y),2))/
      (x*(-1 + y)*y))

fqb1(qh0, qh1, qh2, qbh0, qbh1, qbh2, mh, x, y) = ((-1 + 1/x)*(-((-1 + y)/(-qbh1 + qbh2 + qh1 - qh2)) +
         (-1 + y)/(qh1 - qh2 - (-qbh1 + qbh2)*(-1 + y))) -
      (((-1 + x)*(-1 + y))/
          (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
            qh0*x) - ((-1 + x)*(-1 + y))/
          (-qh2 - qh1*(-1 + x) + qh0*x -
            (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)))/x -
      ((-qbh0 + qh0)*(-1 + x)*(-1 + y))/
       Power(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
         qh0*x,2) + ((-1 + x)*(-1 + y)*
         (qh0 + qbh0*(-1 + y) + Power(mh,2)*y))/
       Power(-qh2 - qh1*(-1 + x) + qh0*x -
         (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y),2))/((-1 + x)*(-1 + y)*y)


fqb2(qh0, qh1, qh2, qbh0, qbh1, qbh2, mh, x, y) = ((-1 + 1/x)*((1 - y)/(qh1 - qh2 - (-qbh1 + qbh2)*(-1 + y)) +
         (-1 + y)/(-qbh1 + qbh2 + qh1 - qh2)) -
      ((1 - y)/
          (-qh2 - qh1*(-1 + x) + qh0*x -
            (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)) +
         (-1 + y)/
          (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
            qh0*x))/x - ((-qbh0 + qh0)*(-1 + y))/
       Power(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
         qh0*x,2) - ((1 - y)*(qh0 + qbh0*(-1 + y) + Power(mh,2)*y))/
       Power(-qh2 - qh1*(-1 + x) + qh0*x -
         (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y),2))/((-1 + y)*y)

function iqh2(qh0, qh1, qh2₀, qbh0, qbh1, qbh2, mh, x, y, atol)
    ok, qh2, it, normf0 = findroot(z -> fq2(qh0, qh1, z, qbh0, qbh1, qbh2, mh, x, y) - 1, qh2₀, NewtonMethod(atol=atol))
    ok || normf0 < 1e-10 || @warn("iqh2 failed: qh2=$qh2, it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    # !ok && (qh2 = qh2₀)
    return ok, qh2
end

function iqbh2(qb2, qh0, qh1, qh2, qbh0, qbh1, qbh2₀, mh, x, y, atol)
    ok, qh2, it, normf0 = findroot(z -> fqb2(qh0, qh1, qh2, qbh0, qbh1, z, mh, x, y) - qb2, qbh2₀, NewtonMethod(atol=atol))
    ok || normf0 < 1e-10 || @warn("iqh2 failed: qh2=$qh2, it=$it, normf0=$normf0")
    ok = normf0 < 1e-5
    # !ok && (qh2 = qh2₀)
    return ok, qh2
end

function ix_fun(op::OrderParams, ep::ExtParams, x)
    @extract ep: β p
    @extract op: q0 q1 qb0 qb1 qb2 qh0 qh1 qh2 qbh0 qbh1 qbh2 y m mh

    return (-(x*((qh0 - qh1 - (-qbh0 + qbh1)*(-1 + y))/
          (-qh2 - qh1*(-1 + x) + qh0*x -
            (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)) +
         ((-qbh0 + qbh1 + qh0 - qh1)*(-1 + y))/
          (qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
            qh0*x))) + Power(x,2)*
     (((-qbh0 + qh0)*(-qbh0 + qbh1 + qh0 - qh1)*(-1 + y))/
        Power(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) - qbh0*x +
          qh0*x,2) - (-(q0*qh0) + q1*qh1 - qb0*qbh0*(-1 + y) +
          qb1*qbh1*(-1 + y))*y -
       ((qh0 - qh1 - (-qbh0 + qbh1)*(-1 + y))*
          (qh0 + qbh0*(-1 + y) + Power(mh,2)*y))/
        Power(-qh2 - qh1*(-1 + x) + qh0*x -
          (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y),2) +
       ((-Power(q0,p) + Power(q1,p) - Power(qb0,p)*(-1 + y) +
            Power(qb1,p)*(-1 + y))*y*Power(β,2))/2.) -
    (-1 + y)*Log(-qbh1 + qbh2 + qh1 - qh2) +
    (-1 + y)*Log(qbh2 - qh2 + qbh1*(-1 + x) - qh1*(-1 + x) -
       qbh0*x + qh0*x) - Log(qh1 - qh2 - (-qbh1 + qbh2)*(-1 + y)) +
    Log(-qh2 - qh1*(-1 + x) + qh0*x -
      (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)))
end

function ix(op::OrderParams, ep::ExtParams, x0, atol)
    ok, x, it, normf0 = findroot(x->ix_fun(op, ep, x), x0, NewtonMethod(atol=atol))
    ok || normf0 < 1e-10 || @warn("ix failed: x=$x, it=$it, normf0=$normf0")
    # (ok && normf0 < 1e-10 && x <= 1+1e-10) || (x=1; @warn("fix x=1"))
    ok = normf0 < 1e-5
    return ok, x
end

fm(qh0, qh1, qh2, qbh0, qbh1, qbh2, mh, x, y) = mh/(-qh2 - qh1*(-1 + x) + qh0*x -
      (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y))
fmh(β, r, k, m) = r*β*m^(k-1)
ifmh(qh0, qh1, qh2, qbh0, qbh1, qbh2, m, x, y) = (true, m*(-qh2 - qh1*(-1 + x) + qh0*x -
      (qbh2 + qbh1*(-1 + x) - qbh0*x)*(-1 + y)))

function converge!(op::OrderParams, ep::ExtParams, pars::Params; fixm=false, fixx=false, fixd=false)
    @extract pars : maxiters verb ϵ ψ

    Δ = Inf
    ok = false

    it = 0
    for it = 1:maxiters
        Δ = 0.0
        verb > 1 && println("it=$it")
        @update  op.qh0      fqh0        Δ ψ verb  ep.β ep.p op.q0
        @update  op.qh1      fqh1        Δ ψ verb  ep.β ep.p op.q1
        @update  op.qbh0     fqbh0       Δ ψ verb  ep.β ep.p op.qb0
        @update  op.qbh1     fqbh1       Δ ψ verb  ep.β ep.p op.qb1

        if fixd
            @updateI  op.qbh2  ok  iqbh2     Δ ψ verb  op.qb2 op.qh0 op.qh1 op.qh2 op.qbh0 op.qbh1 op.qbh2 op.mh op.x op.y 1e-10
        else
            @update  op.qbh2     fqbh2      Δ ψ verb  ep.β ep.p ep.g op.qb2
        end

        @updateI op.qh2 ok  iqh2     Δ ψ verb   op.qh0 op.qh1 op.qh2 op.qbh0 op.qbh1 op.qbh2 op.mh op.x op.y 1e-10
        if fixm
            @updateI  op.mh   ok   ifmh     Δ ψ verb  op.qh0 op.qh1 op.qh2 op.qbh0 op.qbh1 op.qbh2 op.m op.x op.y
        else
            @update  op.mh      fmh     Δ ψ verb  ep.β ep.r ep.k op.m
        end
    

        @update  op.q0      fq0     Δ ψ verb  op.qh0 op.qh1 op.qh2 op.qbh0 op.qbh1 op.qbh2 op.mh op.x op.y
        @update  op.q1      fq1     Δ ψ verb  op.qh0 op.qh1 op.qh2 op.qbh0 op.qbh1 op.qbh2 op.mh op.x op.y
        @update  op.qb0     fqb0    Δ ψ verb  op.qh0 op.qh1 op.qh2 op.qbh0 op.qbh1 op.qbh2 op.mh op.x op.y
        @update  op.qb1     fqb1    Δ ψ verb  op.qh0 op.qh1 op.qh2 op.qbh0 op.qbh1 op.qbh2 op.mh op.x op.y
        if !fixd
            @update  op.qb2     fqb2    Δ ψ verb  op.qh0 op.qh1 op.qh2 op.qbh0 op.qbh1 op.qbh2 op.mh op.x op.y
        end
        if !fixm
            @update  op.m    fm     Δ ψ verb  op.qh0 op.qh1 op.qh2 op.qbh0 op.qbh1 op.qbh2 op.mh op.x op.y
        end
        if !fixx
            @updateI op.x ok  ix      Δ ψ verb  op ep op.x 1e-14
        end


        verb > 1 && println(" Δ=$Δ\n")

        @assert isfinite(Δ)
        ok = Δ < ϵ
        ok && break
    end

    if verb > 0
        println(ok ? "converged" : "failed", " (it=$it Δ=$Δ)")
        println(op)
    end

    return ok
end

Td(p) = √(p*(p-2)^(p-2) / (2(p-1)^(p-1)))

function converge(;
            β=0.5, p=3,
            q0=0.1, q1=0.5, qb0=0.05, qb1=0.4, qb2=0.8,
            qh0=0.2, qh1=0.5, qh2=0.6, qbh0=0.2, qbh1=0.4, qbh2=0.5,
            x = 1., y=2.,
            m=0., mh=0.,
            r = 0., k = 3, g=0., #field strength, field exponent, γ^2/λt
            ϵ=1e-7, maxiters=100000,verb=2, ψ=0.,
            fixm = false, fixx=false
            )
    op = OrderParams(q0, q1, qb0, qb1, qb2, qh0, qh1, qh2, qbh0, qbh1, qbh2, m, mh, x, y)
    ep = ExtParams(β, p, r, k, g)
    pars = Params(ϵ, ψ, maxiters, verb)
    converge!(op, ep, pars, fixm = fixm, fixx=fixx)
    tf = all_therm_func(op, ep)
    println(tf)
    return op, ep, pars, tf
end


function span(;
    p=3,
    q0=0.1, q1=0.5, qb0=0.05, qb1=0.4, qb2=0.8,
    qh0=0.2, qh1=0.5, qh2=0.6, qbh0=0.2, qbh1=0.4, qbh2=0.5,
    m=0, β=1.6, y=1.000001, g=0., x = 1.0001,
    mh=0.1,
    r = 1., k = 3, #field strength, field exponent, γ^2/λt
    ϵ=1e-7, maxiters=10000,verb=2, ψ=0.2,
    kws...)

    op = OrderParams(q0, q1, qb0, qb1, first(qb2), qh0, qh1, qh2, qbh0, qbh1, qbh2, first(m), mh, first(x), first(y))
    ep = ExtParams(first(β), p, r, k, first(g))
    pars = Params(ϵ, ψ, maxiters, verb)
    return span!(op, ep, pars; m=m,y=y,β=β, g=g,x=x,qb2=qb2, kws...)
end

function span!(op::OrderParams, ep::ExtParams, pars::Params;
                m=0, β=1.6, y=1:0.1:5, g=0., x = 0.1, qb2 = 0.8,
                resfile = "results.txt", 
                fixx=false, fixd=false, fixm=true, 
                xlessthan1=true)
    # if !isfile(resfile)
        open(resfile, "w") do f
            allheadersshow(f, ExtParams, OrderParams, ThermFunc)
        end
    # end

    results = []
    for m in m, y in y, β in β, g in g, x in x, qb2 in qb2
        op.m = m; op.y = y; ep.β = β; ep.g = g; 
        fixx && (op.x = x)
        fixd && (op.qb2 = qb2)
        
        opold, epold = deepcopy(op), deepcopy(ep)
        ok = converge!(op, ep, pars; fixm=fixm, fixx=fixx, fixd=fixd)
        tf = all_therm_func(op, ep)
        tf.Σ < -1e-7 && @warn("Sigma negative")
        op.x > 1+1e-6 && @warn("x > 1: $(op.x)")
        if op.x > 1  && xlessthan1
            op, ep = opold, epold
            op.x = 1.0000001
            ok = converge!(op, ep, pars; fixm=fixm, fixx=true, fixd=fixd)
            tf = all_therm_func(op, ep)
        end
        push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))
        # op.x > 1 &&  @warn("β=$β x=$(op.x) ok=$ok")
        if ok
            open(resfile, "a") do rf
                println(rf, plainshow(ep), " ", plainshow(op), " ", plainshow(tf))
            end
        end
        ok || @warn("!ok")
        pars.verb > 0 && print(ep, "\n", tf,"\n")
    end
    return results
end
# #
# function spanx(;
#     p=3,
#     q0=0.1, q1=0.5, qb0=0.05, qb1=0.4, qb2=0.8,
#     qh0=0.2, qh1=0.5, qh2=0.6, qbh0=0.2, qbh1=0.4, qbh2=0.5,
#     m=0:0.1:1, β=3:-0.01:1, y=1:0.1:5, g=0:0.01:1,
#     mh=0.1,
#     x = 0.1:0.01:2,
#     r = 1., k = 3, #field strength, field exponent, γ^2/λt
#     ϵ=1e-7, maxiters=10000,verb=2, ψ=0.2,
#     kws...)

#     op = OrderParams(q0, q1, qb0, qb1, qb2, qh0, qh1, qh2, qbh0, qbh1, qbh2, first(m), mh, first(x), first(y))
#     ep = ExtParams(first(β), p, r, k, first(g))
#     pars = Params(ϵ, ψ, maxiters, verb)
#     return spanx!(op, ep, pars; m=m,y=y,β=β, g=g, x=x, kws...)
# end


# function spanx!(op::OrderParams, ep::ExtParams, pars::Params;
#                 m=0:0.01:1, β=3:-0.01:1, y=1:0.1:5, g=0:0.1:1, x = 0.1:0.01:2,
#                 resfile = "res_spanx.txt")

#     if !isfile(resfile)
#         open(resfile, "w") do f
#             allheadersshow(f, ExtParams, OrderParams, ThermFunc)
#         end
#     end

#     results = []
#     for m in m, y in y, β in β, g in g, x in x
#         op.m = m; op.y = y; ep.β = β; ep.g = g; op.x = x
#         opold, epold = deepcopy(op), deepcopy(ep)
#         ok = converge!(op, ep, pars; fixm=true, fixx=true)
#         tf = all_therm_func(op, ep)
#         push!(results, (ok, deepcopy(ep), deepcopy(op), deepcopy(tf)))
#         if ok
#             open(resfile, "a") do rf
#                 println(rf, plainshow(ep), " ", plainshow(op), " ", plainshow(tf))
#             end
#         end
#         if  !ok
#             @warn("!ok")
#             op, ep = opold, epold
#         end
#         pars.verb > 0 && print(ep, "\n", tf)

#     end
#     return results
# end

function readparams(file, line = 0)
    res = readdlm(file)
    line = line <= 0 ? size(res,1) + line : line-1 # -1 since readdlm discards the header
    op = OrderParams(res[line,6:20]...)
    ep = ExtParams(res[line,1:5]...)
    tf = ThermFunc(res[line,21:end]...)
    return ep, op, tf
end
