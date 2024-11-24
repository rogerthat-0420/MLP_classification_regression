# evaluating the performance of the modelon test dataset

import os
import sys
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt
import joblib

module_path_mlp_regression = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp'))
sys.path.append(module_path_mlp_regression)
from mlp_regression import MLP_Regression

housing=pd.read_csv("../../data/interim/housing_modified.csv")

X=housing.iloc[:, :-1].values
Y=housing.iloc[:, -1].values

np.random.seed(42)

indices = np.random.permutation(len(X))
train_split = int(len(X) * 0.8)

X_train = X[indices[:train_split]]
Y_train = Y[indices[:train_split]].reshape(-1,1)

X_test = X[indices[train_split:]]
Y_test = Y[indices[train_split:]].reshape(-1,1)

indices=np.random.permutation(len(X_test))
test_split = int(len(X_test) * 0.2)

X_val = X_test[indices[:test_split]]
Y_val = Y_test[indices[:test_split]].reshape(-1,1)

X_test = X_test[indices[test_split:]]
Y_test = Y_test[indices[test_split:]].reshape(-1,1)


best_model=joblib.load('../../models/mlp/3_3_mlp_regression_housing_best_model.joblib')  
predictions = best_model.predict(X_test)
true=Y_test

test_loss_mse=best_model.mse_loss(predictions,true)
test_loss_rmse=best_model.rmse_loss(predictions,true)
test_loss_rsquared=best_model.rsquared(predictions,true)
test_loss_mae= np.mean(np.abs(predictions-true))

print("Mean Squared Error: ", test_loss_mse)
print("Root Mean Squared Error: ", test_loss_rmse)
print("R Squared: ", test_loss_rsquared)
print("Mean Absolute Error: ", test_loss_mae)

