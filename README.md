# Generalize Approximate Survey Propagation (GASP) for Phase Retrieval

Julia implementation of the GASP algorithm for the phase retrieval problem from the [paper](https://proceedings.mlr.press/v97/lucibello19a.html)
```
Carlo Lucibello, Luca Saglietti, Yue Lu 
Proceedings of the 36th International Conference on Machine Learning (ICML 2019).
```

# Usage Examples

```julia
include("../../src/common.jl") 
include("../../src/solvers/ASP-UNSAT.jl")

# Define the problem
N = 100
α = 8.0
seed = 13
prob = ASP.Problem("gle"; act=abs, N, α, seed)

# Solve the problem
df, asp, ok = ASP.solve(prob, verb=1, epochs=100)

``` 
The returned dataframe contains usefult information on the training dynamics:
```julia
julia> df
59×5 DataFrame
 Row │ epoch  train_loss   test_loss    ρ           xnorm    
     │ Int64  Float64      Float64      Float64     Float64  
─────┼───────────────────────────────────────────────────────
   1 │     0  0.767821     0.629946     -0.0509776  1.09671
   2 │     1  0.455034     0.35135      -0.122101   0.602033
   3 │     2  0.464739     0.35329      -0.291155   0.482688
   4 │     3  0.347748     0.299792     -0.63928    0.8554
   5 │     4  0.231428     0.244798     -0.870675   1.07874
   6 │     5  0.189828     0.186459     -0.934233   1.14597
   7 │     6  0.0982322    0.0799435    -0.958156   1.04478
   8 │     7  0.0479129    0.0389655    -0.969609   0.922566
   9 │     8  0.034655     0.0329984    -0.976331   0.861052
  10 │     9  0.0204083    0.0213912    -0.985416   0.853163
  11 │    10  0.0100372    0.00988468   -0.99263    0.866157
  ⋮  │   ⋮         ⋮            ⋮           ⋮          ⋮
  49 │    48  3.47078e-12  3.25979e-12  -1.0        0.869989
  50 │    49  2.80336e-12  2.30306e-12  -1.0        0.869989
  51 │    50  1.35705e-12  1.18383e-12  -1.0        0.869989
  52 │    51  5.80518e-13  5.11347e-13  -1.0        0.869989
  53 │    52  4.33221e-13  3.98039e-13  -1.0        0.869989
  54 │    53  2.59281e-13  2.81284e-13  -1.0        0.869989
  55 │    54  1.37262e-13  1.32413e-13  -1.0        0.869989
  56 │    55  7.20638e-14  5.48441e-14  -1.0        0.869989
  57 │    56  3.44351e-14  3.69838e-14  -1.0        0.869989
  58 │    57  2.50171e-14  2.95523e-14  -1.0        0.869989
  59 │    58  1.60428e-14  1.49794e-14  -1.0        0.869989
                                              37 rows omitted
```

See the `test/` folder for more examples.

# Bibliography

```bibtex
@InProceedings{pmlr-v97-lucibello19a,
  title = 	 {Generalized Approximate Survey Propagation for High-Dimensional Estimation},
  author =       {Lucibello, Carlo and Saglietti, Luca and Lu, Yue},
  booktitle = 	 {Proceedings of the 36th International Conference on Machine Learning},
  pages = 	 {4173--4182},
  year = 	 {2019},
  editor = 	 {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
  volume = 	 {97},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {09--15 Jun},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v97/lucibello19a/lucibello19a.pdf},
  url = 	 {https://proceedings.mlr.press/v97/lucibello19a.html},
  abstract = 	 {In Generalized Linear Estimation (GLE) problems, we seek to estimate a signal that is observed through a linear transform followed by a component-wise, possibly nonlinear and noisy, channel. In the Bayesian optimal setting, Generalized Approximate Message Passing (GAMP) is known to achieve optimal performance for GLE. However, its performance can significantly deteriorate whenever there is a mismatch between the assumed and the true generative model, a situation frequently encountered in practice. In this paper, we propose a new algorithm, named Generalized Approximate Survey Propagation (GASP), for solving GLE in the presence of prior or model misspecifications. As a prototypical example, we consider the phase retrieval problem, where we show that GASP outperforms the corresponding GAMP, reducing the reconstruction threshold and, for certain choices of its parameters, approaching Bayesian optimal performance. Furthermore, we present a set of state evolution equations that can precisely characterize the performance of GASP in the high-dimensional limit.}
}
```