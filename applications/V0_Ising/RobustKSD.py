#==================================================
# Library Import
#==================================================

import math
import argparse
import numpy as np
import pandas as pd

import torch
import torch.autograd as autograd

import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.infer import MCMC, NUTS, HMC, Importance, EmpiricalMarginal, Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
from pyro.contrib import autoguide

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.lines as mlines
import seaborn as sns



#==========================================================================
# Parse options
#==========================================================================

parser = argparse.ArgumentParser()
parser.add_argument('--core', default=1, type=int)
parser.add_argument('--theta', default=5, type=int)
parser.add_argument('--data', default=1000, type=int)
parser.add_argument('--beta', default=1, type=float)
parser.add_argument('--thres', default=90, type=int)
parser.add_argument('--noise', default=0.0, type=float)
parser.add_argument('--pv', default=1.0, type=float)
parser.add_argument('--pnum', default=1000, type=int)
args = parser.parse_args()

File_ID = 'RobustKSD_theta=' + str(args.theta) + '_data=' + str(args.data) + '_beta=' + str(args.beta) + '_thres=' + str(args.thres) + '_noise=' + str(args.noise) + '_pv=' + str(args.pv) + '_pnum=' + str(args.pnum)

print(File_ID)


#=========================================================================
# Set hyper-seeds
#==========================================================================

pyro.set_rng_seed(0)
torch.set_num_threads(args.core)




#==================================================
# Define: Differenciation Operators
#==================================================

# X: N x dim (sample number x input dimension)
# X: N x dim x dim (sample number x input dimension x gradient dimension)
def StateShiftPlus(X, state, diff):
    X_shift = ( X.unsqueeze(-1) + torch.eye(X.shape[1])*diff ).transpose(1, 2)
    return torch.where((X_shift==state[-1]+diff), state[0], X_shift) 


# X: N x dim (sample number x input dimension)
# X: N x dim x dim (sample number x input dimension x gradient dimension)
def StateShiftMinus(X, state, diff):
    X_shift = ( X.unsqueeze(-1) - torch.eye(X.shape[1])*diff ).transpose(1, 2)
    return torch.where((X_shift==state[0]-diff), state[-1], X_shift) 



#==================================================
# Deinfe: Model
#==================================================

class Model():
    
    def __init__(self, dim=10, T=5):
        super(Model, self).__init__()
        
        self.dim = dim
        self.T = T
        
        self.cross_edge = self.generate_edge_mat(dim)
        self.cross_edge = self.cross_edge + self.cross_edge.t()
        
        self.p_dim = 1
        self.P_mat = torch.zeros(dim*dim, dim*dim)
        self.P_mat[self.cross_edge==1] = 1
    
    
    def load_data(self, num=None):
        File = "{T:02d}".format(T=self.T)
        X = torch.Tensor(np.genfromtxt('./Dat/ising-samples-n10000-d100-T'+File+'.csv', delimiter=','))
        if num == None:
            return X
        else:
            X_dat = X[0:num]
            if args.noise != 0.0:
                Index = torch.from_numpy(np.random.choice(range(0,num), int(num*args.noise), replace=False))
                X_dat[Index,:] = torch.ones(Index.size(0), 100)
            return X_dat
    
    
    def set_StatX(self, X):
        self.StatX = - 2 * X * ( X @ self.P_mat )
    
    
    def score(self, param):
        return 1 - torch.exp( self.StatX / param )
    
    
    def generate_edge_mat(self, dim):
        padmat = torch.zeros(dim, dim, dim, dim)
        for i in range(dim):
            for j in range(dim):
                if not i - 1 == -1:
                    padmat[i][j][i-1, j] = 1
                if not j - 1 == -1:
                    padmat[i][j][i, j-1] = 1
                if not i + 1 == dim:
                    padmat[i][j][i+1, j] = 1
                if not j + 1 == dim:
                    padmat[i][j][i, j+1] = 1
        return torch.triu(padmat.reshape(dim*dim, dim*dim)).t()



#==================================================
# Define: KSD-Bayes Posterior 
#==================================================

class KSD_Bayes():
    
    def __init__(self, score, log_prior, beta=1, gamma=1, thres=100):
        super(KSD_Bayes, self).__init__()
        self.score = score
        self.log_prior = log_prior
        self.beta = beta
        self.gamma = gamma
        self.thres = thres
    
    
    def HK(self, X):
        M = ( 1 / ( 1 + torch.exp(-( self.thres - torch.sum(X, dim=1).abs() )) ) ).reshape(X.shape[0], 1)
        K = torch.exp( - torch.cdist(X, X, p=0) / X.shape[1] * self.gamma )
        return M * K * M.t()
    
    
    def set_KX(self, X):
        self.X_num = X.shape[0]
        self.HK_XX = self.HK(X)
        self.HK_D_XX = self.HK_D(X)
    
    
    def HK_D(self, X):
        M = ( 1 / ( 1 + torch.exp(-( self.thres - torch.sum(X, dim=1).abs() )) ) )
        
        X_m = StateShiftMinus(X, torch.Tensor([-1,1]), 2)
        M_m = ( 1 / ( 1 + torch.exp(-( self.thres - torch.sum(X_m, dim=2).abs() )) ) )
        K_m = torch.exp( - torch.cdist(X, X_m, p=0) / X.shape[1] )
        
        HK_h = ( self.HK_XX ).unsqueeze(-1)
        HK_m = ( M.reshape(X.shape[0], 1, 1) * K_m * M_m.unsqueeze(0) )
        
        return ( HK_h - HK_m ).mean(axis=1)
    
    
    def ksd(self, param):
        SX = self.score(param)
        T1 = ( SX @ SX.t() * self.HK_XX ).mean()
        T2 = ( SX * self.HK_D_XX ).sum(axis=1).mean()
        return T1 - 2.0 * T2
    
    
    def log_potential(self, param):
        return - self.X_num * self.beta * self.ksd(param) + self.log_prior(param)



#==================================================
# Instantiate: Model and KSD-Bayes
#==================================================

model_ising = Model(dim=10, T=args.theta)
X = model_ising.load_data(args.data)
model_ising.set_StatX(X)

prior = torch.distributions.HalfNormal(args.pv)
ksd_bayes = KSD_Bayes(model_ising.score, prior.log_prob, beta=args.beta, thres=args.thres)
ksd_bayes.set_KX(X)



#==================================================
# Plot Setting
#==================================================

def ax_setting(ax):
    ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%1.1f'))
    
    framewidth = 2.0
    ax.spines["top"].set_linewidth(framewidth)
    ax.spines["left"].set_linewidth(framewidth)
    ax.spines["right"].set_linewidth(framewidth)
    ax.spines["bottom"].set_linewidth(framewidth)
    
    ax.yaxis.set_major_locator(plt.MultipleLocator(2.0))
    ax.tick_params(axis='x', labelsize=28)
    ax.tick_params(axis='y', labelsize=28)

    
def save_fig(p, title):
    fig, ax = plt.subplots(figsize=(7,4))
    ax_setting(ax)
    sns.distplot(p, kde=True)
    fig.tight_layout()
    fig.savefig('./Fig/' + title + '.png')
    plt.close(fig)



#==================================================
# Compute
#==================================================

def negative_log_potential(args):
    return - ksd_bayes.log_potential(args['theta'])

nuts = NUTS(potential_fn=negative_log_potential)
mcmc = MCMC(kernel=nuts, warmup_steps=args.pnum, initial_params={'theta': prior.sample()}, num_samples=args.pnum)
mcmc.run()
post_sample = mcmc.get_samples()['theta']
save_fig(post_sample, File_ID)
np.savetxt("./Res/"+File_ID+".csv", post_sample.numpy(), delimiter=",")


