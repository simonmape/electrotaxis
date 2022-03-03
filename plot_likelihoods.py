import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt

""""
Want to get here: 
* For each summary statistic, the marginal profile for each parameter
* For each summary statistic, the pairwise profile for pairs of parameters
* For the total loss, marginal profile for each parameter 
                                (special case of 1 with last summary stat) 
* For the total loss, the pairwise profile for pairs of parameters
"""

summary_stats = ['le', 'te', 'tbe', 'bs', 'bd', 'ls']
parameters = ['cE', 'beta', 'w', 'u']
parameter_ranges = [np.unique(pl[par].to_numpy()) for par in parameters]

for i, summary_stat in tqdm(enumerate(summary_stats)):
    for j, parameter in enumerate(parameters):
        #Marginal profiles here
        profiles = [pl.loc[pl[parameter] == val].min() for val in parameter_ranges[j]]
        marginal = np.array([profile[summary_stat] for profile in profiles])
        fName = parameter+'_'+summary_stat+'.txt'
        np.savetxt(fName,marginal)
        for k, parameter2 in enumerate(parameters):
            #Pairwise profiles here
            numPar1 = parameter_ranges[j].shape[0]
            numPar2 = parameter_ranges[k].shape[0]
            pairwise = np.zeros((numPar1,numPar2))

            for p1 in range(numPar1):
                for p2 in range(numPar2):
                    par1 = parameter_ranges[j][p1]
                    par2 = parameter_ranges[k][p2]
                    pairwise[p1,p2] = pl.loc[(pl[parameter] == par1) & (pl[parameter2] == par2)].min()[summary_stat]

            fName = parameter+'_'+parameter2+'_'+summary_stat+'.txt'
            np.savetxt(fName,pairwise)