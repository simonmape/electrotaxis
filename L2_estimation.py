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

leading_stim_stat = np.loadtxt('sumstats/leading_stim_stat_smooth.txt')
trailing_stim_stat = np.loadtxt('sumstats/trailing_stim_stat_smooth.txt')
top_stim_stat = np.loadtxt('sumstats/top_stim_stat_smooth.txt')
directionality_stim_stat = np.loadtxt('sumstats/directionality_stim_stat_smooth.txt')
speed_stim_stat = np.loadtxt('sumstats/speed_stim_stat_smooth.txt')

# summary[i, 0] = leading edge
# summary[i, 1] = trailing edge
# summary[i, 2]+summary[i,3] = top and bottom zone
# summary[i, 4] = directionality
# summary[i, 5] = speed

def compute_log_likelihood(summary):
    #Leading edge
    obs = leading_stim_stat
    obs = obs[obs[:,0]<10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*100).astype(int)
    timepoints = timepoints[timepoints<1000]
    sim = summary[timepoints, 0] - summary[timepoints,5]
    le = np.linalg.norm(sim-obs[:,1])

    #Trailing edge
    obs = trailing_stim_stat
    obs = obs[obs[:, 0] < 10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*100).astype(int)
    timepoints = timepoints[timepoints < 1000]
    sim = summary[timepoints, 1] - summary[timepoints,5]
    te = np.linalg.norm(sim-obs[:,1])

    #Top/bottom edge
    obs = top_stim_stat
    obs = obs[obs[:, 0] < 10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*100).astype(int)
    timepoints = timepoints[timepoints < 1000]
    sim = 0.5*(summary[timepoints, 2]+summary[timepoints, 3])
    tbe = np.linalg.norm(sim-obs[:,1])

    # Bulk directionality
    obs = directionality_stim_stat
    obs = obs[obs[:, 0] < 10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*100).astype(int)
    timepoints = timepoints[timepoints < 1000]
    sim = summary[timepoints, 4]
    bd = np.linalg.norm(sim-obs[:,1])

    #Bulk speed
    obs = speed_stim_stat
    obs = obs[obs[:, 0] < 10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*100).astype(int)
    timepoints = timepoints[timepoints < 1000]
    sim = summary[timepoints, 5]
    bs = np.linalg.norm(sim-obs[:,1])

    #Likelihood sum
    ls = le+te+tbe+bd+bs
    return le, te, tbe, bd, bs, ls

for sumstat in tqdm(os.listdir('sumstats/')):
    if re.search(r'\d', sumstat):
        #Read in parameter values used
        param_list = re.split(r'[GCbwu]',sumstat.replace('.txt',''))[2:6]
        param_list = [item.replace('_','.') for item in param_list]
        cE = float(param_list[0][:-1])
        beta = float(param_list[1][:-1])
        w = float(param_list[2][:-1])
        u = float(param_list[3])

        #Read the summary statistic
        summary = np.loadtxt('sumstats/'+sumstat)

        #Compute log-likelihood for this sample
        le, te, tbe, bd, bs, ls = compute_log_likelihood(summary)

        #Record log-likelihoood for this parameter value
        likelihoods.append([cE, beta, w, u, le, te, tbe, bd, bs, ls])

df = pd.DataFrame(likelihoods, columns = ['cE', 'beta', 'w', 'u', 'le', 'te', 'tbe', 'bs', 'bd', 'ls'])
df.to_csv('simple_electrotaxis_L2.csv')