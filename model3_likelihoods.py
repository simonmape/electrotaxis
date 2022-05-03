import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
from scipy.stats import norm
from scipy.ndimage import gaussian_filter1d
""""
Likelihood estimation for model 1 (only propulsion by 
self-advection along direction of polarity and no
superposition/sensitivity differences between different
phenotypes
 """
likelihoods = []

leading_edge_obs = np.loadtxt('leading_stim_stat_smooth.txt')
trailing_edge_obs = np.loadtxt('trailing_stim_stat_smooth.txt')
top_edge_obs = np.loadtxt('top_stim_stat_smooth.txt')
bulk_dir_obs = np.loadtxt('directionality_stim_stat_smooth.txt')
bulk_speed_obs = np.loadtxt('speed_stim_stat_smooth.txt')

def compute_log_likelihood(summary):
    #Leading edge
    obs = leading_edge_obs
    obs = obs[obs[:,0]<10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*40).astype(int)
    timepoints = timepoints[timepoints<1000]
    sim = 0.5*gaussian_filter1d(summary[timepoints, 0] + summary[timepoints, 1],5)
    le = -sum(norm.logpdf(sim, obs[:, 1], obs[:, 2]))

    #Trailing edge
    obs = trailing_edge_obs
    obs = obs[obs[:, 0] < 10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*40).astype(int)
    timepoints = timepoints[timepoints < 1000]
    sim = 0.5*gaussian_filter1d(summary[timepoints, 2] + summary[timepoints, 3],5)
    te = -sum(norm.logpdf(sim, obs[:, 1], obs[:, 2]))

    #Top/bottom edge
    obs = top_edge_obs
    obs = obs[obs[:, 0] < 10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*40).astype(int)
    timepoints = timepoints[timepoints < 1000]
    sim = 0.5*gaussian_filter1d(0.5*(summary[timepoints, 4]+summary[timepoints, 5]),5)
    tbe = -sum(norm.logpdf(sim, obs[:, 1], obs[:, 2]))

    # Bulk directionality
    obs = bulk_dir_obs
    obs = obs[obs[:, 0] < 10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*40).astype(int)
    timepoints = timepoints[timepoints < 1000]
    sim = gaussian_filter1d(summary[timepoints, 6],5)
    bd = -sum(norm.logpdf(sim, obs[:, 1], obs[:, 2]))

    #Bulk speed
    obs = bulk_speed_obs
    obs = obs[obs[:, 0] < 10]
    obs = obs[~np.isnan(obs).any(axis=1)]
    timepoints = np.round(obs[:, 0]*40).astype(int)
    timepoints = timepoints[timepoints < 1000]
    sim = 0.5*gaussian_filter1d(summary[timepoints, 9],5)
    bs = -sum(norm.logpdf(sim, obs[:, 1], obs[:, 2]))

    #Likelihood sum
    ls = le+te+tbe+bd+bs
    return le, te, tbe, bd, bs, ls

for sumstat in tqdm(os.listdir('Model3_Outputs/')):
    if re.search(r'\d', sumstat):
        #Read in parameter values used from file name
        reduced = sumstat.replace('_f','*').replace('model3_cE','').replace('.txt','')
        reduced = reduced.replace('_','.')
        params = reduced.split('*')
        cE = float(params[0])
        f = float(params[1])
        #Read the summary statistic
        summary = np.loadtxt('Model3_Outputs/'+sumstat)
        #Compute log-likelihood for this sample
        le, te, tbe, bd, bs, ls = compute_log_likelihood(summary)
        #Record log-likelihoood for this parameter value
        likelihoods.append([cE,f,le, te, tbe, bd, bs, ls])

df = pd.DataFrame(likelihoods, columns = ['cE','f','le', 'te', 'tbe', 'bs', 'bd', 'ls'])
df.to_csv('model3_loglikelihoods.csv')