import numpy as np
import os

for file in os.listdir(os.getcwd()):
    summary = np.loadtxt(file)
    print(file, 'has', sum(np.all(summary, axis=1)), 'nonzero rows')
