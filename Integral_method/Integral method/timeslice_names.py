# -*- coding: utf-8 -*-
"""
Created on Thu Feb 18 08:19:05 2021

@author: Riccardo Novo
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')

import numpy as np


# %% Inputs

n_seasons = 365
n_daytypes = 1
n_dailytimbrackets = 24


# %% Calculations

tms = []
sss = np.arange(1,n_seasons+1)
ddd = np.arange(1,n_daytypes+1)
ttt = np.arange(1,n_dailytimbrackets+1)

for ss in sss:
    for dd in ddd:
        for tt in ttt:
            name = 'S'+str(ss)+'D'+str(dd)+'T'+str(tt)
            tms.append(name)
