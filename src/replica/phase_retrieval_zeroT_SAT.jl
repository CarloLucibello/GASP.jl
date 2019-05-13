module PhaseRetr

using LittleScienceTools.Roots
using QuadGK
# using FastGaussianQuadratures
include("../common.jl")


###### INTEGRATION  ######
const ∞ = 12.0
const dx = 0.5

const interval = map(x->sign(x)*abs(x)^2, -1:dx:1) .* ∞

∫D(f, int=interval) = quadgk(z->begin
        r = G(z) * f(z)
        isfinite(r) ? r : 0.0
    end, int..., abstol=1e-6, reltol=1e-6, maxevals=10^3)[1]

∫d(f, a, b) = quadgk(f, union([a:dx:b;],b)..., abstol=1e-11, maxevals=10^5)[1]

let s = Dict{Int,Tuple{Vector{Float64},Vector{Float64}}}()
    global gw
    gw(n::Int) = get!(s, n) do
        (x,w) = gausshermite(n)
        return (map(Float64, x * √big(2.0)), map(Float64, w / √(big(π))))
    end
end

function ∫DD(f; n=8)
    (xs, ws) = gw(n)
    s = 0.0
    for (x,w) in zip(xs, ws)
        y = f(x)
        s += w  * ifelse(isfinite(y), y, 0.0)
    end
    return s
end

function deriv(f::Function, i::Integer, x...; δ::Float64 = 1e-5)
    f0 = f(x[1:i-1]..., x[i]-δ, x[i+1:end]...)
    f1 = f(x[1:i-1]..., x[i]+δ, x[i+1:end]...)
    return (f1-f0) / 2δ
end

############### PARAMS

type OrderParams
    q00::Float64
    q0::Float64
    q1::Float64
    q2::Float64
    δq::Float64
    m::Float64
    q̂00::Float64
    p̂::Float64
    q̂0::Float64
    q̂1::Float64
    q̂2::Float64
    δq̂::Float64
end

type ExtParams
    α::Float64
    p::Float64
end

type Params
    ϵ::Float64
    ψ::Float64
    maxiters::Int
    verb::Int
end

type ThermFunc
    ϕ::Float64
end

Base.show(io::IO, op::OrderParams) = shortshow(io, op)
Base.show(io::IO, ep::ExtParams) = shortshow(io, ep)
Base.show(io::IO, params::Params) = shortshow(io, params)
Base.show(io::IO, tf::ThermFunc) = shortshow(io, tf)

####################

### not used for the time being, eventually remove
function findzero_interp(f, x0; dx = 0.005, maxiter = 10, parallel = false)
    s = x0
    ok = false
    iter = 1
    normf0 = Inf
    while !ok && iter <= maxiter
        println("# TRIAL $iter for findzero_interp")
        xmax = s + 2*dx
        xmin  = s - 2*dx
        r = collect(xmin:dx:xmax)
        dummyfile = string("dummy" ,string(rand()),".dat")
        if parallel
            refs = RemoteRef[]
            for i=1:length(r)
                push!(refs, @spawn f(r[i]))
            end

            f0 = [fetch(refs[i]) for i=1:length(r)]
        else
            f0 = [f(r[i]) for i=1:length(r)]
        end

        rf = open(dummyfile, "w")
        for i=1:length(r)
            println(rf, "$(r[i]) $(f0[i])")
        end
        close(rf)
        try
            s = float(readall(`gnuplot -e "filename='$dummyfile'" interpolation.gpl `))
        catch
            println("ERROR GNUPLOT")
            exit()
        end
        run(`rm $dummyfile`)

        normf0 = abs(f(s))
        if normf0 < 1e-5
            println("# SUCCESS x* =$(s), normf0=$normf0")
            ok =true
        else
            warn("failed:x* =$(s), normf0=$normf0")
            ok =false
        end

        if !ok && f0[1]*f0[end] > 0
            dx *= 2
            println("# dx=$dx")
        elseif !ok && f0[1]*f0[end] < 0
            dx /= 2
            println("# dx=$dx")
        end

        iter += 1
    end
    return ok, s, iter, normf0
end

###################################################################################

#### INITIAL TERM ####

Gi(m, p, q0, q1, p̂, q̂0, q̂1, q̂2) = -0.5 * (2 * p * p̂ + q̂2 + (m - 1) * q1 * q̂1 - m * q0 * q̂0)

∂m_Gi(q0, q1, q̂0, q̂1) = -0.5 * (q1 * q̂1 - q0 * q̂0)

#### ENTROPIC TERM ####

Gs(m, p̂, q̂0, q̂1, q̂2) = 0.5 * ((q̂0 + p̂^2)/(q̂1 - q̂2 + m * (q̂0 - q̂1)) + (1-m) /m * log(q̂1 - q̂2) - 1/m * log(q̂1 - q̂2 + m * (q̂0 - q̂1)))

∂m_Gs(m, p̂, q̂0, q̂1, q̂2) = 0.5 * (((q̂0 + p̂^2)*(q̂0 - q̂1)/(m * (q̂0 - q̂1) + q̂1 - q̂2) - 1/m)*(q̂1 - q̂0)/(m * (q̂0 - q̂1) + q̂1 - q̂2) + (log(m * (q̂0 - q̂1) + q̂1 - q̂2) - log(-q̂2 + q̂1)) / m^2 )
∂p̂_Gs(m, p̂, q̂0, q̂1, q̂2) = p̂ / (m * (q̂0 - q̂1) + q̂1 - q̂2)
∂q̂0_Gs(m, p̂, q̂0, q̂1, q̂2) = -(m * (p̂^2 + q̂0))/(2 * (m * (q̂0 - q̂1) + q̂1 - q̂2)^2)
∂q̂1_Gs(m, p̂, q̂0, q̂1, q̂2) = 0.5 * ((1 - m)/(m * (q̂1 - q̂2)) - (1 - m) /(m * (q̂0 - q̂1) + q̂1 - q̂2) * ((p̂^2 + q̂0)/(m * (q̂0 - q̂1) + q̂1 - q̂2) + 1/m))
∂q̂2_Gs(m, p̂, q̂0, q̂1, q̂2) = 0.5 * (-(1 - m)/(m * (q̂1 - q̂2)) + (p̂^2 + q̂0)/(m * (q̂0 - q̂1) + q̂1 - q̂2)^2 + 1/(m * (m * (q̂0 - q̂1) + q̂1 - q̂2)))

# δ = 1e-9
# ∂m_Gs(m, p̂, q̂0, q̂1, q̂2) = (Gs(m+δ, p̂, q̂0, q̂1, q̂2) - Gs(m, p̂, q̂0, q̂1, q̂2)) / δ
# ∂p̂_Gs(m, p̂, q̂0, q̂1, q̂2) = (Gs(m, p̂+δ, q̂0, q̂1, q̂2) - Gs(m, p̂, q̂0, q̂1, q̂2)) / δ
# ∂q̂0_Gs(m, p̂, q̂0, q̂1, q̂2) = (Gs(m, p̂, q̂0+δ, q̂1, q̂2) - Gs(m, p̂, q̂0, q̂1, q̂2)) / δ
# ∂q̂1_Gs(m, p̂, q̂0, q̂1, q̂2) = (Gs(m, p̂, q̂0, q̂1+δ, q̂2) - Gs(m, p̂, q̂0, q̂1, q̂2)) / δ
# ∂q̂2_Gs(m, p̂, q̂0, q̂1, q̂2) = (Gs(m, p̂, q̂0, q̂1, q̂2+δ) - Gs(m, p̂, q̂0, q̂1, q̂2)) / δ

#### ENERGETIC TERM ####

argGe(y, q0, q1, z0, z1) = 0.5 * 1/sqrt(2π * y *(1-q1)) * sum(σ-> exp(-0.5 * (sqrt(q1 -q0) * z1 + sqrt(q0) * z0 + σ * sqrt(y))^2 / (1-q1)), [-1,1])
fy(q00, p, q0, u0, z0) = (sqrt(q00 - p^2 / q0) * u0 + p / sqrt(q0) * z0)^2
function Ge(m, q00, p, q0, q1)
    ∫D(z0->begin
        ∫D(u0->begin
            y = fy(q00, p, q0, u0, z0)
            log(∫D(z1->begin
                (argGe(y, q0, q1, z0, z1))^m
            end))
        end)
    end) / m
end

function ∂all_Ge(m, q00, p, q0, q1)
    ge_0 = Ge(m, q00, p, q0, q1)
    δ = 1e-7

    dq0 = (Ge(m, q00, p, q0+δ, q1)-ge_0) / δ
    dq1 = (Ge(m, q00, p, q0, q1+δ)-ge_0) / δ
    return dq0, dq1
end

function ∂m_Ge(m, q00, p, q0, q1)
    return 1 / m * (∫D(z0->begin
        ∫D(u0->begin
            y = fy(q00, p, q0, u0, z0)
            num = ∫D(z1->begin argGe(y, q0, q1, z0, z1)^m * log(argGe(y, q0, q1, z0, z1)) end)
            den = ∫D(z1->begin argGe(y, q0, q1, z0, z1)^m end)
            num / den
            end)
        end) - Ge(m, q00, p, q0, q1))
end

#####################################

fmz(α, m, q00, p, q0, q1, p̂, q̂0, q̂1, q̂2) = ∂m_Gi(q0, q1, q̂0, q̂1) + ∂m_Gs(m, p̂, q̂0, q̂1, q̂2) + α * ∂m_Ge(m, q00, p, q0, q1)

function imz(α, m, q00, p, q0, q1, p̂, q̂0, q̂1, q̂2; verb = 1, ϵ = 1e-6)
    m0 = m
    ok, m, it, normf0 = findroot(m->fmz(α, m, q00, p, q0, q1, p̂, q̂0, q̂1, q̂2), m0, NewtonMethod(atol=ϵ))
    # try
        # ok, m, it, normf0 = findzero_interp(m->fmz(α, m, q00, p, q0, q1, p̂, q̂0, q̂1, q̂2), m0, maxiter = 1, dx = 0.1)
    #     ok || normf0 < 1e-6 || warn("imz failed: x=$x, it=$it, normf0=$normf0")
    #     if !ok || normf0 > 1e-5
    #         x = x0
    #     end
    # catch
    #     x = x0
    # end
    return m
end

############ Thermodynamic functions

free_entropy(α, m, q00, p, q0, q1, p̂, q̂0, q̂1, q̂2) = Gi(m, p, q0, q1, p̂, q̂0, q̂1, q̂2) + Gs(m, p̂, q̂0, q̂1, q̂2) + α * Ge(m, q00, p, q0, q1)

###########################

#### SADDLE POINT EQUATIONS ####
#
# q00 = 1
# q0 = -2/m * ∂q̂0_Gs
# q1 = 2/(m-1) * ∂q̂1_Gs
# q2 = 1
# q̂00 = -1
#
# q̂0 = -2α/m * ∂q0_Ge
# q̂1 = 2α/(m-1) * ∂q1_Ge
# 0 = -1/2 +  ∂q̂2_Gs --->>> q̂2
# 0 = ∂p̂_Gs - p --->>> p  or  p̂


function fallhats(α, m, q00, p, q0, q1)
	dq0, dq1 = ∂all_Ge(m, q00, p, q0, q1)

    q̂0 = - 2 * α / m * dq0
    q̂1 = 2 * α / (m-1) * dq1
    return q̂0, q̂1
end

function fall(m, p̂, q̂0, q̂1, q̂2)
    dq0 = ∂q̂0_Gs(m, p̂, q̂0, q̂1, q̂2)
    dq1 = ∂q̂1_Gs(m, p̂, q̂0, q̂1, q̂2)

    q0 = -2/m * dq0
    q1 = 2/(m-1) * dq1
    return q0, q1
end

fp̂(m, p, p̂, q̂0, q̂1, q̂2) = ∂p̂_Gs(m, p̂, q̂0, q̂1, q̂2) - p
fq̂2(m, p, p̂, q̂0, q̂1, q̂2) = ∂q̂2_Gs(m, p̂, q̂0, q̂1, q̂2) - 0.5

function ip̂(m, p, p̂, q̂0, q̂1, q̂2; verb = 0, ϵ = 1e-8)
    p̂0 = p̂
    ok, p̂, it, normf0 = findroot(x->fp̂(m, p, x, q̂0, q̂1, q̂2), p̂0, NewtonMethod(atol=ϵ))
    #ok, M, it, normf0 = findzero_interp(M->∂_Ge(5, Q, q0, q1, δq, M, x, K, avgξ, varξ, f′), M0, dx=0.1)

    ok || warn("ip̂ failed: p̂=$(p̂), it=$it, normf0=$normf0")
    if !ok || normf0 > 1e-5
        return p̂0
    end
    return p̂
end

function iq̂2(m, p, p̂, q̂0, q̂1, q̂2; verb = 0, ϵ = 1e-8)
    q̂20 = q̂2
    ok, q̂2, it, normf0 = findroot(x->fq̂2(m, p, p̂, q̂0, q̂1, x), q̂20, NewtonMethod(atol=ϵ))
    #ok, M, it, normf0 = findzero_interp(M->∂_Ge(5, Q, q0, q1, δq, M, x, K, avgξ, varξ, f′), M0, dx=0.1)

    ok || warn("iq̂2 failed: q̂2=$(q̂2), it=$it, normf0=$normf0")
    if !ok || normf0 > 1e-5
        return q̂20
    end
    return q̂2
end

function fix_inequalities_hat(m, p, q0, q1, p̂, q̂0, q̂1, q̂2)
    ok = false
    t = 0
    while !ok
        t += 1
        ok = true
        if q̂1 < q̂2
            q̂1 += 1e-3
            ok = false
        end
        if q̂1 - q̂2 + m * (q̂0 - q̂1) < 0
            q̂1 += 1e-3
            ok = false
        end
    end
    t > 1 && println("***fixed***")
    return m, p, q0, q1, p̂, q̂0, q̂1, q̂2
end

function fix_inequalities_nonhat(m, p, q0, q1, p̂, q̂0, q̂1, q̂2)
    ok = false
    t = 0
    while !ok
        ok = true
        t += 1
        if q0 < 0
            q0 = rand() * 1e-5
            ok = false
        end
        if q1 < 0
            q1 = rand() * 1e-5
            ok = false
        end
        if q1 + 1e-9 > 1
            q1 -= 1e-9
            ok = false
        end
        if q1 < q0 + 1e-9
            # q0 -= 1e-5
            q1 += 1e-5
            ok = false
        end
        if p^2 / q0 > 1 + 1e-5
            q0 += 1e-5
            ok = false
        end
    end
    t > 1 && println("***fixed***")
    return m, p, q0, q1, p̂, q̂0, q̂1, q̂2
end

function converge!(op::OrderParams, ep::ExtParams, pars::Params; updatem = true)

    Δ = Inf
    ok = false

    fixstep = 1e-8
    minfixstep = 1e-10

    old_op = OrderParams[]

    op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2 = fix_inequalities_hat(op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2)
    op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2 = fix_inequalities_nonhat(op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2)

    it = 0
    for it = 1:pars.maxiters

        Δ = 0.0
        fixed = false
        if pars.verb > 1
            println("it=$it")
            println(op)
            println(ep)
            println(pars)
        end

        @time q̂0, q̂1 = fallhats(ep.α, op.m, op.q00, ep.p, op.q0, op.q1)

        @update op.q̂2  iq̂2    Δ pars.ψ pars.verb   op.m ep.p op.p̂ op.q̂0 op.q̂1 op.q̂2
        @update op.p̂  ip̂    Δ pars.ψ pars.verb    op.m ep.p op.p̂ op.q̂0 op.q̂1 op.q̂2
        # @update ep.p      identity      Δ pars.ψ pars.verb  ∂p̂_Gs(op.m, op.p̂, op.q̂0, op.q̂1, op.q̂2)
        @update op.q̂0     identity       Δ pars.ψ pars.verb   q̂0
        @update op.q̂1     identity       Δ pars.ψ pars.verb   q̂1



        q0, q1 = fall(op.m, op.p̂, op.q̂0, op.q̂1, op.q̂2)

        @update op.q0     identity       Δ pars.ψ pars.verb q0
        @update op.q1     identity       Δ pars.ψ pars.verb q1

        op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2 = fix_inequalities_nonhat(op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2)



        op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2 = fix_inequalities_hat(op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2)


        if updatem
            @update op.m  imz    Δ pars.ψ pars.verb ep.α op.m op.q00 ep.p op.q0 op.q1 op.p̂ op.q̂0 op.q̂1 op.q̂2
            if op.m < 0
                op.m = 0.01
            end
        end

        # if it % 10 == 1
        #     println("complexity ", fmz(ep.α, op.m, op.q00, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2))
        #     tf.ϕ = free_entropy(op.m, op.q00, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2)
        #     println(tf)
        # end

        pars.verb > 1 && println(" Δ=$Δ", fixed ? " [*]" : "", "\n")

        ok = Δ < pars.ϵ && (!fixed || fixstep < minfixstep)

        if Δ < pars.ϵ && fixed && fixstep ≥ minfixstep
            fixstep /= 2
            pars.verb > 1 && info("new fixstep=$fixstep")
        end

        if ok
            println("complexity ", fmz(ep.α, op.m, op.q00, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2))
            tf.ϕ = free_entropy(ep.α, op.m, op.q00, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2)
            println(tf)
            break
		end
    end

    if pars.verb > 0
        println(ok ? "converged" : "failed", " (it=$it Δ=$Δ)")
        println("    ", ep)
        println("    ", op)
    end

    return ok
end

function initialize_op(q00=1.0,q0=.99,q1=.991,q2=1.0,δq=0.0,m=0.9999,q̂00=-1.0,
    p̂=1.,q̂0=-2.2522734568265226e-5,q̂1=7.033975269406627e-8,q̂2=-1.0101021798951892,δq̂=0.0)

    op = OrderParams(q00, q0, q1, q2, δq, m, q̂00, p̂, q̂0, q̂1, q̂2, δq̂)
    return op
end


function span(; lstα = [0.6], lstp = [0.6], op = nothing,
                ϵ = 1e-5, ψ = 0.1, maxiters = 500, verb = 4,
                resfile = nothing, updatem = true)

    default_resfile = "results_phase_retrieval_zeroT_SAT.txt"
    resfile == nothing && (resfile = default_resfile)

    pars = Params(ϵ, ψ, maxiters, verb)
    results = Any[]

    lockfile = "reslock.tmp"

    op == nothing && (op = initialize_op())
    ep = ExtParams(lstα[1], lstp[1])
    println(op)

    for α in lstα, p in lstp
        ep.α = α
        ep.p = p

        function seek!(m)
            op.m = m
            converge!(op, ep, pars, updatem = false)
            return fmz(ep.α, op.m, op.q00, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2)
        end

        if !updatem
            ok, op.m, it, norm = findroot(seek!, op.m, NewtonMethod(atol=ϵ))
        else
            converge!(op, ep, pars, updatem = true)
        end

        tf = ThermFunc(free_entropy(op.m, ep.p, op.q0, op.q1, op.p̂, op.q̂0, op.q̂1, op.q̂2))

        push!(results, (ok, deepcopy(op), deepcopy(ep), deepcopy(tf)))
        verb > 0 && println(tf)
        println("\n#####  NEW ITER  ###############\n")
        if ok
            exclusive(lockfile) do
                open(resfile, "a") do rf
                    println(rf, plainshow(ep), " ", plainshow(tf), " ", plainshow(op))
                end
            end
        end
        ok || break
    end
    return results, op
end

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
