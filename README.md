# KSD-Bayes
Experiment Code for "Robust Generalised Bayesian Inference for Intractable Likelihoods" (https://arxiv.org/abs/2104.07359)

=================

This repository contains Matlab (R2019b) code to reproduce the experiments in "Robust Generalised Bayesian Inference for Intractable Likelihoods".

The main function is KSD_Bayes, which performs conjugate inference for exponential family models using KSD-Bayes. 

There are four applications considered in the paper: (1) A Gaussian location model, (2) a two-dimensional scale parameter estimation problem due to Liu et al, (3) density estimation using a kernel exponential family model, and (4) an exponential graphical model. Results for each application can be reproduced by running the files reproduce_[X].m, where [X] = Gauss, Liu, KEF, EGM.