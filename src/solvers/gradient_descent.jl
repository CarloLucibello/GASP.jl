module GD
using Optimisers
include("../common.jl")

# predict(x, A) = abs2.(A'x)
predict(x, A) = abs.(A'x)

loss(x, A, y) = MSE(predict(x, A), y)

function train!(x, data, opt; backtrack=false)
    Δ = 0.
    for (A, y) in data
        l, grad = withgradient(x -> loss(x, A, y), x)
        g = grad[1]
        # remove_projection!(g, x)
        Optimisers.update!(opt, x, g)
        # project_sphere!(x)
        Δ += vecnorm(g) / length(g)
        if backtrack && loss(x, A, y) >  l
            update!(x, -g, opt)
            # project_sphere!(x)
            opt.lr /= 2
            @warn("backtracking: new lr $(opt.lr)")
        end
    end
    Δ
end

function initx(x₀, prob)
    @extract prob: A y
    if x₀ isa Vector
        x = deepcopy(x₀)
    elseif x₀ == :spectral
        x = spectral_init_optimal(A, y)
    else
        x = randn(size(A,1)); project_sphere!(x)
    end
    @assert length(x) == size(A, 1)
    x
end


function create_s_at_ρ(t, ρ, ϵ)
    N = length(t)
    r = randn(length(t))
    l = 10
    η = 0.1
    loss(η) = (t' * (η*t + (1-η)*r)/(vecnorm(t)*vecnorm((η*t + (1-η)*r))) - ρ)^2
    while l > ϵ
        l, g = withgradient(loss, η)
        η -= 0.1 * g[1]
    end
    return project_sphere!(η*t + (1-η)*r)
end


function solve(; N=1000,α=1.5, seedp=-1, kws...)
    prob = Problem("gle", N=N, α=α, act=abs2, seed=seedp)
    solve(prob; kws...)
end

function reg_abs(x, β)
    ifelse(x > 1/β, abs(x), β * x^2)
end

function htanh(x, β)
    ifelse(x > 1/β, sign(x), 2β*x)
end

function dhtanh(x, β)
    ifelse(x > 1/β, 0, 2β)
end


function solve(problem;
            seed = -1,
            lr = 1,
            epochs = 200,
            infotime = 1,  # report every `infotime` epochs
            x₀ = :spectral, # [:spectral, nothing, a configuration]
            batchsize = -1,
            verb = 2,
            backtrack=false, # lower the lr if neeeded to have non increasing steps
            ϵ = 1e-10  # stopping criterion for the norm of the gradient
        )

    @extract problem: A y Atst ytst teacher


    seed > 0 && srand(seed)
    N, M = size(A)
    verb > 1 && info("# N=$N, M=$M, α=$(M/N)")
    @assert length(y) == M
    batchsize <= 0 && (batchsize = M)

    x = initx(x₀, problem)
    opt = Optimisers.setup(Descent(0.1), x)

    df = DataFrame(epoch = Int[],
            train_loss = Float64[],
            test_loss = Float64[],
            ρ = Float64[])

    report(epoch, Δ, verb) = begin
            res = (epoch=epoch,
                    train_loss = loss(x, A, y),
                    test_loss = loss(x, Atst, ytst),
                    ρ = abs(overlap(x, teacher.x0)))

            push!(df, res)
            verb > 1 &&  (cprint(res); cprintln(:Δ=>Δ));
        end

    epoch = 0
    Δ = 1.
    report(epoch, Δ, verb+1);
    stats = @timed while epoch < epochs
        epoch += 1
        data = batches(A, y, batchsize; shuffle=true)
        Δ = train!(x, data, opt; backtrack)
        epoch % infotime == 0 && report(epoch, Δ, verb)
        Δ < ϵ && break
    end
    report(epoch, Δ, verb+1)
    verb > 0 && Δ > ϵ && @warn("not converged!")

    return problem, x, df
end

end #module
