# 3.6 Analysis of the Model

import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

module_path_mlp_regression = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp'))
sys.path.append(module_path_mlp_regression)
from mlp_regression import MLP_Regression

diabetes=pd.read_csv("../../data/external/diabetes.csv")

features = diabetes.iloc[:, :-1] 
target = diabetes.iloc[:, -1]   

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
diabetes_scaled = pd.concat([scaled_features_df, target.reset_index(drop=True)], axis=1)

X = diabetes_scaled.iloc[:, :-1].values
Y = diabetes_scaled.iloc[:, -1].values
X_check=diabetes_scaled.iloc[:, :-1]

np.random.seed(42)
indices = np.random.permutation(X.shape[0])

train_split = int(0.8 * X.shape[0])

X_train = X[indices[:train_split]]
Y_train = Y[indices[:train_split]].reshape(-1,1)
X_train_check=X_check.iloc[indices[:train_split]]

X_val = X[indices[train_split:]]    
Y_val = Y[indices[train_split:]].reshape(-1,1)
X_val_check=X_check.iloc[indices[train_split:]]

val_test_split = int(len(X_val) * 0.8)

X_test = X_val[val_test_split:]
Y_test = Y_val[val_test_split:]
X_test_check=X_val_check.iloc[val_test_split:]

X_val = X_val[:val_test_split]
Y_val = Y_val[:val_test_split]
X_val_check=X_val_check.iloc[:val_test_split]


mlp=joblib.load('../../models/mlp/3_5_logistic_mse_sgd.joblib')  

# logs = []

# # LLM ***
# for i, point in enumerate(X_test):
#     point = point.reshape(1, -1)
#     prediction = mlp.predict(point)
#     mlp.init_layers(point.shape[0])
#     pred_loss=mlp.forward_pass(point)
#     loss = mlp.loss_func(pred_loss, Y_test[i].reshape(1, -1))
#     x_test_check_row = X_test_check.iloc[i].to_dict()
#     logs.append({**x_test_check_row, "Y_test": Y_test[i], "Prediction": prediction[0][0], "MSE Loss": loss})

# log_df = pd.DataFrame(logs)
# log_df_sorted = log_df.sort_values(by="MSE Loss", ascending=True)
# print(log_df_sorted)
# # LLM ***


logs = []

# LLM ***
for i, point in enumerate(X_test):
    point = point.reshape(1, -1)
    prediction = mlp.predict(point)
    mlp.init_layers(point.shape[0])
    pred_loss=mlp.forward_pass(point)
    loss = mlp.loss_func(pred_loss, Y[i].reshape(1, -1))
    x_test_check_row = X_test_check.iloc[i].to_dict()
    logs.append({**x_test_check_row, "Y_test": Y[i], "Prediction": prediction[0][0], "MSE Loss": loss})

log_df = pd.DataFrame(logs)
log_df_sorted = log_df.sort_values(by="MSE Loss", ascending=True)
print(log_df_sorted)


import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Normalize the features to [0, 1] range to better compare on a single plot
# scaler = MinMaxScaler()
# X_normalized = scaler.fit_transform(log_df_sorted[X_test_check.columns])

# # Convert back to DataFrame for easier plotting
# normalized_df = pd.DataFrame(X_normalized, columns=X_test_check.columns)

# Plotting


num_features = len(log_df_sorted.columns)
selected_features = log_df_sorted.columns[:8]

# Create a grid of subplots, with one subplot for each feature
fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(21, 21))

# Loop through each feature and plot it against the MSE Loss on different subplots
for i, feature in enumerate(selected_features):
    ax=axes[i // 3, i % 3]
    sns.scatterplot(x=log_df_sorted[feature], y=log_df_sorted["MSE Loss"],ax=ax )

# Adjust layout to avoid overlap
fig.suptitle("Feature vs. MSE Loss for Test dataset")
plt.savefig("figures/3_6_test.png")

