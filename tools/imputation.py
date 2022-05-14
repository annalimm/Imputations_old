

import sys 
sys.path.append('..')
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
from tools import mice_i
import numpy as np
import pandas as pd
import tensorflow as tf
import MIDASpy as md
torch.set_default_tensor_type('torch.DoubleTensor')

def impute(X_miss, imputer_name = 'mf', mode='mae'):
    name = imputer_name

    if name == 'mf':
        imp = SimpleImputer(strategy = 'most_frequent').fit_transform(X_miss)
    elif name == 'mean':
        imp = SimpleImputer().fit_transform(X_miss)
    elif name == 'ice':
        imp = IterativeImputer(max_iter=50, random_state=0, sample_posterior = True, estimator = BayesianRidge()).fit_transform(X_miss)
        # imp = IterativeImputer(max_iter=50, random_state=0, sample_posterior = True).fit_transform(X_miss)
    
    elif name == 'mice':
        mice_imps = []
        for i in range(5): ## ух ты тут за случайность в изначальном заполнении отвечает i в random_state
            imp = IterativeImputer(max_iter = 50, random_state = i, sample_posterior = True, estimator = BayesianRidge()).fit_transform(X_miss)
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
        imp = sk_imp
    elif name == 'mice_i':
        vm = mice_i.VanilaMICE(max_iter=5)
        vmices = []
        mask = np.isnan(X_miss)
        for i in range(5):
            mask_col = np.where(mask)[0]
            mask_row = np.where(mask)[1]
            ids = [[row, col] for row, col in zip(np.where(mask)[0],np.where(mask)[1]) ]
            vmice = vm.transform(pd.DataFrame(X_miss),columns_missing = np.unique(mask_row), nan_ids = ids, iter_id=i)
            vmices.append(vmice)
        vmices = np.array(vmices)
        imp = np.mean(vmices)
    elif name == 'miceforest':
        # Create kernel. 
        imputer_forest = mf.ImputationKernel(
                                        X_miss,
                                        datasets=1,
                                        mean_match_candidates=5,
                                        save_all_iterations=True,
                                        random_state=1991
                                        )
        optimal_parameters, losses = imputer_forest.tune_parameters(
                                        dataset=0,
                                        optimization_steps=5
                                        )
        imputer_forest.mice(iterations=1, boosting='gbdt', min_sum_hessian_in_leaf=0.01)
        imp = imputer_forest.complete_data(dataset=0)
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
        imp, lin_maes, lin_rmses = lin_rr_imputer.fit_transform(X_miss, verbose=True)#, X_true = X_full)
        imp = imp
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

    return imp

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
