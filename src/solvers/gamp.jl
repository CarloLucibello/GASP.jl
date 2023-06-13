

function AMP_phase_retrieval(y, A, kappa; opt...)
    N, M = size(A)
    α = M/N

    μₓ = zeros(N,1)
    varₓ= ones(N,1)
    R = zeros(N,1)
    gμ = zeros(M,1)

    varₘ = zeros(N,1)
    fx = copy(y)
    V = (A.^2)' * σ₂_x


    for t = 1:epochs
        Vnew .= (A.^2)' varₓ
        fx .= ψ * fx + (1-ψ) *  (A' μ_x - Vnew .* gμ)
        V .= ψ * V + (1-ψ) * Vnew

        for i=1:M
            if (Y(i) < kappa)
                gμ[i] = (Y[i]-fx[i])/(1e-10+V[i])
            else
                gμ[i] = (Y[i].*tanh(Y[i].*fx[i]./(1e-10+V[i]))-fx[i])./(1e-10+V[i])
            end
        end

        mean_gμ2 = mean(gμ.^2) * ones(M, 1)
        Σ₂ .= ((A.^2) * mean_gμ2).^(-1)
        R = μ_x + (Σ₂ .* (gμ' * A))

        #Compute the new values a and c O(1)
        μₓ_new = Fun_a(S2,R',rho,exp_gauss,sqrt_var_gauss.^2)';
        varₓ_new = Fun_b(S2,R',rho,exp_gauss,sqrt_var_gauss.^2)';

        #udate final a and b
        μₓ = (1-ψ) * μₓ_new + ψ * μₓ
        varₓ = (1-ψ) * varₓ_new + ψ * varₓ

        # Test of the convergence : Are the averages constant in time or enough
        # accuratly reconstructed?
        if (crit_test <= conv_criterion_t)
            disp('Converged');
            disp(crit_test)
            disp(conv_criterion_t)
            break
        end
        if (max(size(S)) > 2)
            err_true=min(error_estimate(F_a,S),error_estimate(-F_a,S));
            if (err_true <= conv_criterion_acc)
                disp('Solution found');
                break
            end
        end

        #Printing to screen
        if ((affichage > 0)&&(mod(t,affichage) == 0))
            PR=sprintf('%d %e %f',full([t crit_test damp_mes]));
            if (~(max(size(S)) < 2))
                PR2=sprintf(' ->>>> %f', min(sum((F_a-S).^2)/N,sum((F_a+S).^2)/N));
                PR=[PR PR2];
            end
            disp(PR)
        end
        t=t+1;
    end

    if (max(size(S)) > 2)
        disp('Final average error of the reconstruction');
        solution=Fun_solution(S2,R',rho,exp_gauss,sqrt_var_gauss.^2)';
        disp(err_true);
        final_error=err_true;
    end

    X=solution';  mean=exp_gauss; variance=sqrt_var_gauss.^2;

    disp('TESTING THE GENERALIZATION ERROR');
    test_sum2=0;

    for i=1:10000
        newPhi=randn(1,N)/sqrt(N);
        newZ=newPhi*S;
        if newZ<-kappa
            newY=-newZ;
        else
           newY=newZ;
        end
        Z_mean=abs(newPhi*F_a);
        Z_var=sum(F_b)/max(size(F_b));
        expectation=(sqrt(2*Z_var)/sqrt(pi))*exp(-(kappa+Z_mean)^2/(2*Z_var))+0.5*Z_mean*erfc(-Z_mean/sqrt(2*Z_var))-0.5*Z_mean*(1+erf(-Z_mean/sqrt(2*Z_var)));
        test_sum2=test_sum2+(expectation-newY)^2;
    end
    test_sum2/10000


end
