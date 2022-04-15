import numpy as np
import pandas as pd
import torch
import os
import matplotlib.pyplot as plt
from tools.utils import plot_style
plt.style.use(plot_style)
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import scale
import multiprocess as mp
from multiprocessing import Process, Manager, Pool
# from tools.softimpute import softimpute, cv_softimpute
import sys 
sys.path.append('..')
from alternative_imputers.muzellec_imputers import RRimputer
from tools.utils import pick_epsilon, MAE, RMSE #error estimators
from tools.amputation import produce_NA, plot_style
from tqdm import tqdm
from sklearn.datasets import load_boston, load_wine, fetch_california_housing, load_iris, load_diabetes, load_linnerud
from sklearn.preprocessing import StandardScaler
import os
def get_data(dataset_name): 
    scaler = StandardScaler()
    if dataset_name == 'boston':
        X_full, y_full = load_boston(return_X_y=True)
    elif dataset_name == 'california_housing':
        X_full, y_full = fetch_california_housing(return_X_y = True)
    elif dataset_name == 'wine':
        X_full, y_full = load_wine(return_X_y=True)
       
    elif dataset_name == 'iris':
        X_full, y_full = load_iris(return_X_y=True)
    elif dataset_name == 'diabetes':
        X_full, y_full = load_diabetes(return_X_y=True)
        
    if dataset_name == 'real':
        curdir = os.getcwd()
        os.chdir("..")
        X_full = np.genfromtxt(os.path.join('data', 'data_ivanovo.csv') , delimiter=';', skip_header=2)
        df = pd.DataFrame(X_full)
        df.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
        X_full = np.array(df)
        X_full = scaler.fit_transform(X_full)
        os.chdir(curdir)
        return X_full, None
    else:
        X_full = scaler.fit_transform(X_full, y_full)
        return scale(X_full), scale(y_full)

def impute(dataset_name, p_miss, mecha, imputer_name = 'mf', mode='mae'):
    maes = []
    rmses = []
    name = imputer_name
    X_full, y_full = get_data(dataset_name)
    X_miss_t = produce_NA(X_full, p_miss = p_miss, p_obs = 0.1, mecha = mecha)
    X_miss = X_miss_t['X_incomp']
    mask = X_miss_t['mask'] 

    
    if name == 'mf':
        imp = SimpleImputer(strategy = 'most_frequent').fit_transform(X_miss)
    elif name == 'mean':
        imp = SimpleImputer().fit_transform(X_miss)
    elif name == 'ice':
        #imp = IterativeImputer(max_iter=50, random_state=0, sample_posterior = False).fit_transform(X_miss)
        imp = IterativeImputer(max_iter=50, random_state=0, sample_posterior = True).fit_transform(X_miss)
    
    elif name == 'mice':
        mice_imps = []
        for i in range(5): ## ух ты тут за случайность в изначальном заполнении отвечает i в random_state
            imp = IterativeImputer(max_iter = 50, random_state = i, sample_posterior = True).fit_transform(X_miss)
            mice_imps.append(imp)
        imp = sum(mice_imps)/len(mice_imps)

    elif name == 'linearRR':  
        lr = 1e-2
        n, d = X_miss.shape
        epsilon = pick_epsilon(X_miss) # Set the regularization parameter as a multiple of the median distance, as per the paper.
        #Create the imputation models
        d_ = d - 1
        models = {}
        for i in range(d):
            models[i] = torch.nn.Linear(d_, 1)
        #Create the imputer
        lin_rr_imputer = RRimputer(models, eps=epsilon, lr=lr)
        imp, lin_maes, lin_rmses = lin_rr_imputer.fit_transform(X_miss, verbose=True, X_true = torch.from_numpy(X_full))
    
    elif name == 'mice_r':
        imp = mice_R(X_miss, maxit=50, m=5, seed = 1, meth = 'pmm')

    elif name == 'knn':
        imp = KNNImputer().fit_transform(X_miss)

    elif name == 'full':
        imp = X_full

    if mode == 'mae':
        mae = MAE(imp, X_full, mask)
        maes.append(mae)
        mae_std = np.std(np.array(maes))

        return np.mean(maes), mae_std, imp
    elif mode == 'bayesianRidge' and dataset_name != 'real':
         # дальше оцениваем связь в данных,  внашем случае с BayesianRidge, но надо проверить еще и другие метрики 
        br_estimator = BayesianRidge()
        CVS = cross_val_score(br_estimator, imp, y_full, scoring = "neg_mean_squared_error", cv = 5)

        score_full_data = pd.DataFrame(CVS, columns = ["Full Data"], )

        return -float(score_full_data.mean()), -float(score_full_data.std()), imp
    else:# dataset_name == 'real':
        print("No labels for the real data")
        return 0, 0, 0


def run_imputer(params):
    idx, n_impute, dataset, p_miss, mecha, imputer_name, mode = params
    means = []
    imps = []
    for i in range(int(n_impute)):
        mean, std, imp = impute(dataset_name = dataset, p_miss = float(p_miss), mecha = mecha, imputer_name = imputer_name, mode= mode)
        means.append(mean)
        imps.append(imp)
    return [idx, np.mean(means), np.std(means), np.mean(imps)]

if __name__ == '__main__':
    n_impute = 1, 
    datasets = ['iris', 'diabetes', 'california_housing'],
    p_miss = [0.05],#, 0.20, 0.4],
    modes = ['mae'],#, 'bayesianRidge'],
    mechas = ['MCAR'],#, 'MAR', 'MNAR'],
    imputer_names = ['mf', 'mean']#, 'ice', 'mice']#, 'mice_r']
                
    
    params_grid = np.array(np.meshgrid( n_impute, datasets, p_miss, mechas, imputer_names, modes)).T.reshape(-1,6)
    params_grid = np.vstack((np.arange(params_grid.shape[0]), params_grid.T)).T
    
    results = []
    for param in tqdm(params_grid): ## tqdm -- progress bar
        result = run_imputer(param)
        results.append(result)
    
    

