using JLD2
using DataFrames
using LittleScienceTools.Measuring
using Glob: glob
include("ASP-UNSAT.jl")

function multi_run(;
        ## outer loop
        ms = [2,10,20,100],

        ## inner loop
        N = 10000,
        seeds=1:20,
        αs = 1.5:0.01:2,
        λs = 0,
        epochs = 1000,
        dropλ = true,
        x₀ = :nothing, #[:randn, :teacher]
        ρ₀ = 1e-1,
        ϵ = 1e-8,
    )   

    procs=[]
    for m in ms
        julia = "/home/lucibello/Git/julia-0.6/julia"
        args_out = "ms=$m"
        args_in = "N=$N,seeds=$seeds,αs=$(αs),λs=$λs, epochs=$epochs,
                dropλ=$dropλ,x₀=:$(x₀),ρ₀=$(ρ₀),ϵ=$ϵ"
        
        proc = spawn(`$julia -e "
        include(\"run_parallel.jl\")
        BLAS.set_num_threads(2)
        single_run($args_in, $args_out)
        "`)
        
        push!(procs,proc)
    end
    procs
end

function single_run(;
        N = 100,
        seeds = 1:10,
        αs = 2.1,
        ms = [1,2,10,20,100],
        λs = [0,0.001,0.01,0.1],
        epochs = 100,
        dropλ = true,
        x₀ = :randn, #[:randn,:nothing]
        ρ₀ = 0,
        ϵ = 1e-8
    )

    for α in αs, m in ms, λ in λs, s in seeds
        prob = ASP.Problem("gle",N=N,α=α,act=abs,seed=s)
        df, amp, prms, ok = ASP.solve(prob, m=m, ϵ=ϵ, λ=λ, x₀=x₀, ρ₀=ρ₀, dropλ=dropλ, epochs=epochs, verb=0)
        x0 = norm(prob.teacher.x0)
        x = amp.x
        filename = "../../rawdata/ASP-UNSAT/res_$(x₀)rho$(ρ₀)_N$(N)_alpha$(α)_m$(m)_lambda$(λ)_dropl_seed$(s).jld"
        @show filename 
        @save filename df x x0 prms ok
    end
end


function data_analysis(;x₀=:randn)
    obs = ObsTable([:ρ,:N,:α,:m,:λ])
    if x₀ == :randn
        infiles = glob("res_N*_dropl_*.jld", "../../rawdata/ASP-UNSAT")
        append!(infiles, glob("res_randnrho*_N*_dropl_*.jld", "../../rawdata/ASP-UNSAT"))
    else x₀ == :nothing
        infiles = glob("res_initrho*_dropl_*.jld", "../../rawdata/ASP-UNSAT")
        append!(infiles, glob("res_nothingrho*_N*_dropl_*.jld", "../../rawdata/ASP-UNSAT"))
    end

    for fname in infiles
        ## Extract parameters from file name
        namesplts = split(fname[1:end-4], '_')[2:end]
        exclude = ["dropl"]
        vars = []
        seed = -1
        for n in namesplts
            any(x->startswith(n,x), exclude) && continue
            i = findfirst(isnumber, n)
            x = parse(Float64, n[i:end])
            if startswith(n,"seed")
                seed = Int(x)
            else
                push!(vars, isinteger(x) ? Int(x) : x)
            end
        end
        length(vars) < 5 && prepend!(vars,0)

        #### Load file and accumulate statistics ####
        @load fname df x x0 ok
        
        ## Statistics of the first phase (λ>0)
        i0 = findlast(x->x > 0, df[:λ])
        i0 == 0 && (i0 = length(df[:λ])) 
        
        ρ, λ, xnorm = abs(df[:ρ][i0]), df[:λ][i0], df[:xnorm][i0]
        !isfinite(ρ) && (ρ=0)
        !isfinite(xnorm) && (xnorm=0)
        obs[vars][:ρ1] &= ρ
        obs[vars][:xnorm1] &= xnorm
        mse = xnorm^2 + norm(x0)^2/length(x0) - 2ρ
        obs[vars][:MSE1] &= mse
        obs[vars][:Psucc1] &= mse < 1e-2 # weak recovery
        if i0 < 1000
            obs[vars][:t1] &= i0
        end

        ## Statistics of the second phase (λ=0)
        i0 = length(df[:λ])
        ρ, λ, xnorm = abs(df[:ρ][i0]), df[:λ][i0], df[:xnorm][i0]
        !isfinite(ρ) && (ρ=0)
        !isfinite(xnorm) && (xnorm=0)
        obs[vars][:ρ2] &= ρ
        obs[vars][:xnorm2] &= xnorm
        mse = xnorm^2 + norm(x0)^2/length(x0) - 2ρ
        obs[vars][:MSE2] &= mse
        obs[vars][:Psucc2] &= mse < 1e-10 # strong recovery
        if i0 < 1000
            obs[vars][:t2] &= i0
        end
    end
    
    open("res_$(x₀)_ASP.dat","w") do f
        println(f,obs)
    end
        
    obs
end


function data_analysis_traj(;x₀=:randn,
        ## TO SAVE time series
        Nt = 10000,
        αt = 1.7,
        initρt = 1e-3)

    obs_traj = ObsTable([:ρ,:N,:α,:m,:λ,:seed,:t])
    if x₀ == :randn
        infiles = glob("res_N$(Nt)_dropl_*.jld", "../../rawdata/ASP-UNSAT")
        append!(infiles, glob("res_randnrho*_N$(Nt)_*_dropl_*.jld", "../../rawdata/ASP-UNSAT"))
    else x₀ == :nothing
        infiles = glob("res_initrho*_N$(Nt)_*_dropl_*.jld", "../../rawdata/ASP-UNSAT")
        append!(infiles, glob("res_nothingrho*_N$(Nt)_*_dropl_*.jld", "../../rawdata/ASP-UNSAT"))
    end

    for fname in infiles
        ## Extract parameters from file name
        namesplts = split(fname[1:end-4], '_')[2:end]
        exclude = ["dropl"]
        vars = []
        seed = -1
        for n in namesplts
            any(x->startswith(n,x), exclude) && continue
            i = findfirst(isnumber, n)
            x = parse(Float64, n[i:end])
            if startswith(n,"seed")
                seed = Int(x)
            else
                push!(vars, isinteger(x) ? Int(x) : x)
            end
        end
        length(vars) < 5 && prepend!(vars,0)
        (vars[1]==initρt && vars[2]==Nt && vars[3]==αt && vars[5]==0) || continue
        #### Load file and accumulate statistics ####
        @load fname df x x0 ok
        
        # ## Statistics of the first phase (λ>0)
        i0 = findlast(x->x > 0, df[:λ])
        i0 == 0 && (i0 = length(df[:λ])) 
        
        # # ρ, λ, xnorm = abs(df[:ρ][i0]), df[:λ][i0], df[:xnorm][i0]
        # # !isfinite(ρ) && (ρ=0)
        # # !isfinite(xnorm) && (xnorm=0)
        # # obs[vars][:ρ1] &= ρ
        # # obs[vars][:xnorm1] &= xnorm
        # # mse = xnorm^2 + norm(x0)^2/length(x0) - 2ρ
        # # obs[vars][:MSE1] &= mse
        # # obs[vars][:Psucc1] &= mse < 1e-2 # weak recovery
        # # if i0 < 1000
        # #     obs[vars][:t1] &= i0
        # # end

        # ## TIME SERIES
        ts = df[:epoch]
        ρs = df[:ρ]
        xnorms = df[:xnorm]
        for (i,t) in enumerate(ts)
            varst = [vars; [seed,t]]
            o = obs_traj[varst] 
            o[:ρ] &= ρs[i]
            o[:xnorm] &= xnorms[i]
        end
    end

    open("res_$(x₀)_ASP_traj_N$(Nt)_alpha$(αt)_initrho$(initρt).dat","w") do f
        println(f,obs_traj)
    end
        
    obs_traj
end