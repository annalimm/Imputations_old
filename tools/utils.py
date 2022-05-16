import torch
import numpy as np
from cycler import cycler
from scipy import optimize

#packages for R mice:
import pandas as pd
# import rpy2.robjects as ro
# from rpy2.robjects.conversion import localconverter
# from rpy2.robjects import pandas2ri
# torch.set_default_tensor_type('torch.DoubleTensor')
torch.set_default_tensor_type('torch.DoubleTensor')


colors = [ '#EDA98D', '#8DDCED' ,'#93ED8D', '#8D98ED', '#BFFBE2', '#FBBFDB', '#8598BC']
# Matplotlib style definition for saving plots
plot_style = {
    'axes.prop_cycle': cycler(
        'color',
        ['#1f17f4',
        '#ffa40e',
        '#ff3487',
        '#008b00',
        '#17becf',
        '#850085'
        ]
        ) + cycler('marker', ['o', 's', '^', 'v', 'D', 'd']),
    'axes.edgecolor': '0.3',
    'xtick.color': '0.3',
    'ytick.color': '0.3',
    'xtick.labelsize': '15',
    'ytick.labelsize': '15',
    'axes.labelcolor': 'black',
    'axes.grid': True,
    'grid.color': '#E68F6B',
    'grid.alpha': '0.8',
    'grid.linestyle': '--',
    'axes.labelsize':'20',
    'font.size': '15',
    'lines.linewidth': '1',
    'figure.figsize': '12, 6',
    'lines.markeredgewidth': '0',
    'lines.markersize': '2',
    'axes.spines.right': True,
    'axes.spines.top': True,
    'xtick.minor.visible': True,
    'ytick.minor.visible': True,
    'savefig.dpi': '180'
    }



def mice_R(X_miss, maxit=5, m=5, seed=1, meth=None):
    df = pd.DataFrame(np.array(X_miss)).iloc[:, :]
    df.columns = [f"Col{i}" for i in range(df.columns.shape[0])]
    with localconverter(ro.default_converter + pandas2ri.converter):
        df_r = ro.conversion.py2rpy(df)

    r = ro.r
    r['source']('mice.R')
    impute_mice_r = ro.globalenv['impute_mice']
    if meth is None:
        df_result_r = impute_mice_r(df_r, maxit=maxit, m=m, seed=seed)
    else:
        df_result_r = impute_mice_r(df_r, maxit=maxit, m=m, seed=seed, meth=meth)

    with localconverter(ro.default_converter + pandas2ri.converter):
        pd_from_r_df = ro.conversion.rpy2py(df_result_r)
    return np.array(pd_from_r_df)


def nanmean(v, *args, **kwargs):
    """
    A Pytorch version on Numpy's nanmean
    """
    v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


#### Accuracy Metrics ####
def MAE(X, X_true, mask):
    """
    Mean Absolute Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic
    
    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        MAE : float

    """
    # for x in [X, X_true, mask]:
    #     to_torch = torch.is_tensor(x) ## output a pytorch tensor, or a numpy array
    #     if not to_torch:
    #         x = torch.from_numpy(x)
    # mask = torch.from_numpy(mask)
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)
        
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        X_true = torch.from_numpy(X_true)
        return torch.abs(X[mask_] - X_true[mask_]).sum() / mask_.sum()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.absolute(X[mask_] - X_true[mask_]).sum() / mask_.sum()



def RMSE(X, X_true, mask):
    """
    Root Mean Squared Error (MAE) between imputed variables and ground truth. Pytorch/Numpy agnostic

    Parameters
    ----------
    X : torch.DoubleTensor or np.ndarray, shape (n, d)
        Data with imputed variables.

    X_true : torch.DoubleTensor or np.ndarray, shape (n, d)
        Ground truth.

    mask : torch.BoolTensor or np.ndarray of booleans, shape (n, d)
        Missing value mask (missing if True)

    Returns
    -------
        RMSE : float

    """
    # for x in [X, X_true, mask]:
    #     to_torch = torch.is_tensor(x) ## output a pytorch tensor, or a numpy array
    #     if not to_torch:
    #         x = torch.from_numpy(x)
    # to_torch = torch.is_tensor(mask) ## output a pytorch tensor, or a numpy array
    # if not to_torch:
    #     mask = torch.from_numpy(mask)
    to_torch = torch.is_tensor(X) ## output a pytorch tensor, or a numpy array
    if not to_torch:
        X = torch.from_numpy(X)
    if torch.is_tensor(mask):
        mask_ = mask.bool()
        X_true = torch.from_numpy(X_true)
        return (((X[mask_] - X_true[mask_]) ** 2).sum() / mask_.sum()).sqrt()
    else: # should be an ndarray
        mask_ = mask.astype(bool)
        return np.sqrt(((X[mask_] - X_true[mask_])**2).sum() / mask_.sum())




#### Automatic selection of the regularization parameter ####
def pick_epsilon(X, quant=0.5, mult=0.05, max_points=2000):
    """
        Returns a quantile (times a multiplier) of the halved pairwise squared distances in X.
        Used to select a regularization parameter for Sinkhorn distances.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data on which distances will be computed.

    quant : float, default = 0.5
        Quantile to return (default is median).

    mult : float, default = 0.05
        Mutiplier to apply to the quantiles.

    max_points : int, default = 2000
        If the length of X is larger than max_points, estimate the quantile on a random subset of size max_points to
        avoid memory overloads.

    Returns
    -------
        epsilon: float

    """
    means = nanmean(X, 0)
    X_ = X.clone()
    mask = torch.isnan(X_)
    X_[mask] = (mask * means)[mask]

    idx = np.random.choice(len(X_), min(max_points, len(X_)), replace=False)
    X = X_[idx]
    dists = ((X[:, None] - X) ** 2).sum(2).flatten() / 2.
    dists = dists[dists > 0]

    return quantile(dists, quant, 0).item() * mult


#### Quantile ######
def quantile(X, q, dim=None):
    """
    Returns the q-th quantile.

    Parameters
    ----------
    X : torch.DoubleTensor or torch.cuda.DoubleTensor, shape (n, d)
        Input data.

    q : float
        Quantile level (starting from lower values).

    dim : int or None, default = None
        Dimension allong which to compute quantiles. If None, the tensor is flattened and one value is returned.


    Returns
    -------
        quantiles : torch.DoubleTensor

    """
    return X.kthvalue(int(q * len(X)), dim=dim)[0]
