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

directionality_stim_stat = np.loadtxt('directionality_stim_stat.txt')
leading_stim_stat = np.loadtxt('leading_stim_stat.txt')
speed_stim_stat = np.loadtxt('speed_stim_stat.txt')
top_stim_stat = np.loadtxt('top_stim_stat.txt')
trailing_stim_stat = np.loadtxt('trailing_stim_stat.txt')

def compute_log_likelihood(summary):
    #Leading edge

    #Trailing edge

    #Top/bottom edge

    #Bulk speed

    #Bulk directionality

    #Likelihood sum

    return le, te, tbe, bs, bd, ls

for sumstat in tqdm(os.listdir()):
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
    le, te, tbe, bs, bd, ls = compute_log_likelihood(summary)

    #Record log-likelihoood for this parameter value
    likelihoods.append([cE, beta, w, u, le, te, tbe, bs, bd, ls])

df = pd.DataFrame(data, columns = ['cE', 'beta', 'w', 'u', 'le', 'te', 'tbe', 'bs', 'bd', 'ls'])