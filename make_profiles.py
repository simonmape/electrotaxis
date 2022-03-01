import numpy as np
import pandas as pd
import re
import os
from tqdm import tqdm
from scipy.stats import norm

profile_likelihoods = pd.read_csv('simple_electrotaxis_loglikelihoods.csv')

""""
Want to get here: 
* For each summary statistic, the marginal profile for each parameter
* For each summary statistic, the pairwise profile for pairs of parameters
* For the total loss, marginal profile for each parameter 
                                (special case of 1 with last summary stat) 
* For the total loss, the pairwise profile for pairs of parameters
"""


