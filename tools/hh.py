mice_imps = []
for i in range(5):
    imp = IterativeImputer(max_iter = 50, 
    random_state = i, 
    sample_posterior = True, 
    estimator = BayesianRidge()).fit_transform(X_miss)
    mice_imps.append(imp)
imp = sum(mice_imps)/len(mice_imps)