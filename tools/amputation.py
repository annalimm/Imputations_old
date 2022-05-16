#%%
from pyampute.ampute import MultivariateAmputation
import itertools as it
import numpy as np
from pandas import DataFrame

def __ampute(X_obs, 
             p_miss, 
             mech, 
             vars_observed=[0], 
             score_func = "sigmoid-right", 
             weights=None):
    all_vars = [np.array([i]) for i in range(X_obs.shape[1])]
    vars_to_amp = np.delete(all_vars, vars_observed, None)    
    patt =  [{'incomplete_vars': vars_to_amp, "mechanism": mech,
              "score_to_probability_func": score_func,
              "weights": weights}]
    ma = MultivariateAmputation(
        prop = p_miss, 
        patterns=patt
    )
    X_miss = ma.fit_transform(X_obs)
    mask = np.isnan(X_miss)
    return X_miss, mask


def ampute(X_obs, p_miss, mech, vars_observed=[0, 1], score_func="sigmoid-right", weights=None, frame=False):
    if frame:
        columns = X_obs.columns
    X_obs = np.array(X_obs)
    X_miss_0, mask_0 = __ampute(X_obs, p_miss, mech, vars_observed=[vars_observed[0]], score_func=score_func, weights=weights)
    X_miss_1, mask_1 = __ampute(X_obs, p_miss, mech, vars_observed=[vars_observed[1]], score_func=score_func, weights=weights)
    mask = np.vstack((mask_1[:, 0], mask_0[:, 1:].T)).T
    X_miss = np.vstack((X_miss_1[:, 0], X_miss_0[:, 1:].T)).T
    if frame:
        return DataFrame(X_miss, columns = columns), DataFrame(mask, columns = columns)
    else:
        return X_miss, mask
