# 2.4 Evaluating Single-label Classification Model 
import os
import sys
import pandas as pd
import numpy as np   
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

module_path_mlp_classifier = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp'))
sys.path.append(module_path_mlp_classifier)
from mlp_classifier import MLP_Classifier

df_wine=pd.read_csv('../../data/interim/modified_Wine_QT.csv')
X=df_wine.iloc[:, :-1].values
Y=df_wine.iloc[:, -1].values
np.random.seed(40)
indices = np.random.permutation(len(X))
train_split = int(len(X) * 0.7)
val_split = int(len(X) * 0.2)

X_train = X[indices[:train_split]]
Y_train = Y[indices[:train_split]]

X_val = X[indices[train_split:train_split+val_split]]
Y_val = Y[indices[train_split:train_split+val_split]]

X_test = X[indices[train_split+val_split:]]
Y_test = Y[indices[train_split+val_split:]]

best_model=joblib.load('../../models/mlp/2_3_mlp_classification_wineQT_best_model.joblib')  
predictions = best_model.predict(X_test)
predictions_loss=best_model.forward_pass(X_test)
true_loss=np.squeeze(np.eye(len(np.unique(Y_test)))[Y_test.astype(int).reshape(-1)])

test_loss=best_model.loss(predictions_loss,true_loss)
f1_test=f1_score(Y_test, predictions, average='macro')
precision_test=precision_score(Y_test, predictions, average='macro',zero_division=0)
recall_test=recall_score(Y_test, predictions, average='macro',zero_division=0)
accuracy_test=accuracy_score(Y_test, predictions)

print("Test Accuracy: ", accuracy_test)
print("Test Precision: ", precision_test)
print("Test Recall: ", recall_test)
print("Test F1 Score: ", f1_test)
print("Test Loss: ", test_loss)

