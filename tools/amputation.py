#%%
from pyampute.ampute import MultivariateAmputation
import itertools as it
import numpy as np


def ampute(X_obs, p_miss, mech, vars_observed=[0]):
    all_vars = [np.array([i]) for i in range(X_obs.shape[1])]
    vars_to_amp = np.delete(all_vars, vars_observed, None)    
    patt =  [{'incomplete_vars': vars_to_amp, "mechanism": mech}]#, "score_to_probability_func": "sigmoid-right"
    ma = MultivariateAmputation(
        prop = p_miss, 
        patterns=patt
    )
    X_miss = ma.fit_transform(X_obs)
    mask = np.isnan(X_miss)
    return X_miss, mask
