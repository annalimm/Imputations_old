

import sys 
sys.path.append('..')
from tools.amputation import produce_NA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from alternative_imputers.muzellec_imputers import RRimputer
from tools.utils import pick_epsilon, MAE, RMSE #error estimators
import torch
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score


def impute(X_full, p_miss, mecha, imputer_name = 'mf', mode='mae'):
    name = imputer_name
    X_miss_t = produce_NA(X_full, p_miss = p_miss, p_obs = 0.1, mecha = mecha)
    X_miss = X_miss_t['X_incomp']
    mask = X_miss_t['mask'] 

    
    if name == 'mf':
        imp = SimpleImputer(strategy = 'most_frequent').fit_transform(X_miss)
    elif name == 'mean':
        imp = SimpleImputer().fit_transform(X_miss)
    elif name == 'ice':
        imp = IterativeImputer(max_iter=50, random_state=0, sample_posterior = False).fit_transform(X_miss)
        # imp = IterativeImputer(max_iter=50, random_state=0, sample_posterior = True).fit_transform(X_miss)
    
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
    return X_full, X_miss, mask, imp

def assess_impute(X_full, mask, imp, mode = 'mae', y_full = None):
    if mode == 'mae':
        mae = MAE(imp, X_full, mask)
        return mae, 0, imp
    elif mode == 'bayesianRidge':
            # дальше оцениваем связь в данных,  внашем случае с BayesianRidge, но надо проверить еще и другие метрики 
        br_estimator = BayesianRidge()
        CVS = cross_val_score(br_estimator, imp, y_full, scoring = "neg_mean_squared_error", cv = 5)

        score_full_data = pd.DataFrame(CVS, columns = ["Full Data"], )

        return -float(score_full_data.mean()), -float(score_full_data.std()), imp
    else:# dataset_name == 'real':
        print("No such mode")
        return 0, 0, 0
