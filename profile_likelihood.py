import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
from scipy.stats import norm

""""
Parameter grid here is given by Gamma =  2
U ranges from 0 to 20 in 25 increments
v_sa ranges from 2 to 50 in 25 increments
beta ranges from 0.05 to 2.05 in increments of 0.1
cE ranges from 0.05 to 2.05 in incements of 0.1
"""
likelihoods = []

leading_stim_stat = np.loadtxt('leading_stim_stat.txt')
trailing_stim_stat = np.loadtxt('trailing_stim_stat.txt')
top_stim_stat = np.loadtxt('top_stim_stat.txt')
directionality_stim_stat = np.loadtxt('directionality_stim_stat.txt')
speed_stim_stat = np.loadtxt('speed_stim_stat.txt')

# summary[i, 0] = leading edge
# summary[i, 1] = trailing edge
# summary[i, 2]+summary[i,3] = top and bottom zone = assemble((U * v_load[0] + w_sa * p_load[0]) * dx_sub(1)) / area
# summary[i, 4] = directionality
# summary[i, 5] = speed

def compute_log_likelihood(summary):
    #Leading edge
    obs = leading_stim_stat
    timepoints = np.round(obs[:, 0]*100)
    sim = summary[timepoints, 0]
    le = norm.logpdf(sim, obs[:, 1], obs[:, 2])

    #Trailing edge
    obs = trailing_stim_stat
    timepoints = np.round(obs[:, 0]*100)
    sim = summary[timepoints, 1]
    te = norm.logpdf(sim, obs[:, 1], obs[:, 2])

    #Top/bottom edge
    obs = top_stim_stat
    timepoints = np.round(obs[:, 0]*100)
    sim = 0.5*(summary[timepoints, 2]+summary[timepoints, 3])
    tbe = norm.logpdf(sim, obs[:, 1], obs[:, 2])

    # Bulk directionality
    obs = directionality_stim_stat
    timepoints = np.round(obs[:, 0]*100)
    sim = summary[timepoints, 4]
    te = norm.logpdf(sim, obs[:, 1], obs[:, 2])

    #Bulk speed
    obs = speed_stim_stat
    timepoints = np.round(obs[:, 0]*100)
    sim = summary[timepoints, 5]
    te = norm.logpdf(sim, obs[:, 1], obs[:, 2])

    #Likelihood sum
    ls = le+te+tbe+bd+bs

    return le, te, tbe, bd, bs, ls

for sumstat in tqdm(os.listdir('sumstats/')):
    #Read in parameter values used
    param_list = re.split('G|C|b|w|u|',file.replace('.txt',''))[2:6]
    param_list = [item.replace('_','.')[:-1] for item in param_list]
    cE = float(param_list[0])
    beta = float(param_list[1])
    w = float(param_list[2])
    u = float(param_list[3])

    #Read the summary statistic
    summary = np.loadtxt(sumstat)

    #Compute log-likelihood for this sample
    le, te, tbe, bd, bs, ls = compute_log_likelihood(summary)

    #Record log-likelihoood for this parameter value
    likelihoods.append([cE, beta, w, u, le, te, tbe, bd, bs, ls])

df = pd.DataFrame(data, columns = ['cE', 'beta', 'w', 'u', 'le', 'te', 'tbe', 'bs', 'bd', 'ls'])