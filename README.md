# Generalize Approximate Survey Propagation (GASP) for Phase Retrieval

Julia implementation of the GASP algorithm for the phase retrieval problem from the [paper](https://proceedings.mlr.press/v97/lucibello19a.html)
```
Carlo Lucibello, Luca Saglietti, Yue Lu 
Proceedings of the 36th International Conference on Machine Learning (ICML 2019).
```



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