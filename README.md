# Metropolized knockoff sampling

#### Description

This is a collection of scripts in R and Python that sample *exact* knockoffs for graphical models via the method of *Metropolized knockoff sampling* (Metro). Scripts are grouped by the distribution of the original features. They reproduce the results in Bates, Candès, Janson and Wang (2020+), including knockoff sampling for discrete Markov chains, continuous Markov chains (Gaussian, heavy-tailed, and asymmetric), Ising/Gibbs models on grids and Potts model on a graph inspired by a real problem in protein contact prediction. The script for the Potts model implements a general knockoff sampler that works for any general graphical model.

A tutorial can be found [here](http://web.stanford.edu/group/candes/metro). Correspondence should be addressed to [Stephen Bates](https://web.stanford.edu/~stephen6) and [Wenshuo Wang](https://wenshuow.github.io).

#### Reference

Stephen Bates, Emmanuel Candès, Lucas Janson and Wenshuo Wang. (2020+). *Metropolized Knockoff Sampling*. Journal of the American Statistical Association (to appear) [[pdf](http://lucasjanson.fas.harvard.edu/papers/Metropolized_Knockoff_Sampling-Bates_ea-2019.pdf)] [[arXiv](https://arxiv.org/abs/1903.00434)]
