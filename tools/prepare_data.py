from sklearn.datasets import load_boston, load_wine, fetch_california_housing, load_iris, load_diabetes, load_linnerud
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os

def get_data_from_name(dataset_name): 
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