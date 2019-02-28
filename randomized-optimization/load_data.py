import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data_dir = 'data/'

def load_file(fn):
    return pd.read_csv(os.path.join(data_dir, fn), sep=',').values

""" Notes on Abalone data:
- Sex (M, F, I) @ 0
- arr[1-7] are std measurements (continuous)
- arr[8] is rings (indicative of age)
- Label: predict rings
- See dataset distributions here: https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.names
"""
def load_abalone():
    le = LabelEncoder()
    le.fit(['M', 'F', 'I'])
    arr = load_file('abalone.data')
    arr[:,0] = le.transform(arr[:,0]) - 1 # Center
    y = arr[:, 8].astype('ushort')
    # bin these since we're doing classification and not regression (also consider division)
    bins = 8 * np.arange(5)
    y = np.digitize(y, bins)
    x = arr[:,:8].astype('float')
    
    # Normalize all columns
    mu = np.mean(x[:,1:], axis=0)
    std = np.std(x[:,1:], axis=0)
    x[:,1:] = (x[:,1:] - mu) / std
    return x, y

""" Notes on Heart Disease data:
- Most data is just std medical measurements, continuous
- Replaced 0-3 classification of heart disease stage with binary task as suggested on dataset page
"""
def load_heart_disease():
    arr = load_file('heart_disease.data')
    y = np.where(arr[:, -1] > 1, 1, 0) # Binary classification for presence of disease
    x = arr[:,:-1] # Fill missing data with mean
    x = np.where(x == '?', np.nan, x).astype('float')
    nan_mean = np.nanmean(x, axis=0)
    x = np.where(np.isnan(x), nan_mean, x)
    
    # Normalize
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = (x - mu)/std
    return x, y

def load_diabetes():
    arr = load_file("diabetes.csv")
    y = arr[:,8]
    x = arr[:,:8]
    mu = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    x = (x - mu)/std
    return x,y
