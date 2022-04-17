

import sys 
sys.path.append('..')
from tools.amputation import produce_NA
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer, KNNImputer
from alternative_imputers.muzellec_imputers import RRimputer, OTimputer
import miceforest as mf
from tools.utils import pick_epsilon, MAE, RMSE #error estimators
import torch
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import tensorflow as tf
import MIDASpy as md
torch.set_default_tensor_type('torch.DoubleTensor')

def impute(X_full, p_miss, mecha, imputer_name = 'mf', mode='mae', X_miss_t = None):
    name = imputer_name
    if X_miss_t is None:
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
    elif name == 'sinkhorn':
        n, d = X_miss.shape
        batchsize = 128 # If the batch size is larger than half the dataset's size,
                        # it will be redefined in the imputation methods.
        lr = 1e-2
        epsilon = pick_epsilon(X_miss) # Set the regularization parameter as a multiple of the median distance, as per the paper.
        sk_imputer = OTimputer(eps=epsilon, batchsize=batchsize, lr=lr, niter=2000)

        sk_imp, sk_maes, sk_rmses = sk_imputer.fit_transform(X_miss, verbose=True, report_interval=500, X_true=X_full)
        imp = sk_imp.detach().numpy()

    elif name == 'miceforest':
        # Create kernel. 
        kds_gbdt = mf.ImputationKernel(
        X_miss.detach().numpy(),
        datasets=1,
        save_all_iterations=True,
        random_state=1991
        )
        # Using the first ImputationKernel in kernel to tune parameters
        # with the default settings.
        optimal_parameters, losses = kds_gbdt.tune_parameters(
        dataset=0,
        optimization_steps=5
        )
        # We need to add a small minimum hessian, or lightgbm will complain:
        kds_gbdt.mice(iterations=1, boosting='gbdt', min_sum_hessian_in_leaf=0.01)
        # Return the completed kernel data
        imp = kds_gbdt.complete_data(dataset=0)
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
        imp, lin_maes, lin_rmses = lin_rr_imputer.fit_transform(X_miss, verbose=True, X_true = X_full)
        imp = imp.detach().numpy()
        # lin_maes = lin_maes.detach().numpy()
        # lin_rmses = lin_rmses.detach().numpy()
    elif name == 'mice_r':
        imp = mice_R(X_miss, maxit=50, m=5, seed = 1, meth = 'pmm')
    elif name == 'knn':
        imp = KNNImputer().fit_transform(X_miss)
    elif name == 'midas':
        midas_imputer = md.Midas(layer_structure = [256,256], vae_layer = False, seed = 89, input_drop = 0.75)
        midas_imputer.build_model(pd.DataFrame(np.array(X_miss)))#, softmax_columns = cat_cols_list)
        midas_imputer.train_model(training_epochs = 20)
        imp = np.array(midas_imputer.generate_samples(m=1).output_list[0])
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
