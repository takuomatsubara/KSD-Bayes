%% Get Started

This repository contains Matlab (R2021a) code to reproduce the experiments 
in "Robust Generalised Bayesian Inference for Intractable Likelihoods".

The main function is KSD_Bayes, which performs conjugate inference for 
exponential family models using KSD-Bayes. 

There are four applications considered in the paper: 

 (1) A Gaussian location model, 
 (2) a two-dimensional scale parameter estimation problem due to Liu et al, 
 (3) density estimation using a kernel exponential family model, and 
 (4) an exponential graphical model. 

Results for each application can be reproduced by running the files 
reproduce_[X].m, where [X] = Gauss, Liu, KEF, EGM.

%% Dependencies

The "utilities" folder contains routines for regularised covariance 
estimation, from the "RegularizedSCM" toolbox. See:

Esa Ollila and Elias Raninen, "Matlab RegularizedSCM Toolbox Version 1.11
Available online: http://users.spa.aalto.fi/esollila/regscm/, August 2021.

Esa Ollila and Elias Raninen, "Optimal shrinkage covariance matrix 
estimation under random sampling from elliptical distributions," 
arXiv:1808.10188 [stat.ME].