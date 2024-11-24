# 2.6 Evaluating multi-label Classification Model 
import os
import sys
import pandas as pd
import numpy as np   
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, hamming_loss

module_path_mlp_multi_classifier = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp'))
sys.path.append(module_path_mlp_multi_classifier)
from mlp_multi_classifier import MLP_Multi_Classifier

data_adv=pd.read_csv("../../data/interim/advertisement_modified.csv")

X=data_adv.iloc[:, :-8].values
Y=data_adv.iloc[:, -8:].values

np.random.seed(42)

indices = np.random.permutation(len(X))
train_split = int(len(X) * 0.7)
val_split = int(len(X) * 0.2)

X_train = X[indices[:train_split]]
Y_train = Y[indices[:train_split]]

X_val = X[indices[train_split:train_split+val_split]]
Y_val = Y[indices[train_split:train_split+val_split]]

X_test = X[indices[train_split+val_split:]]
Y_test = Y[indices[train_split+val_split:]]

best_model=joblib.load('../../models/mlp/2_6_mlp_classification_advertisement_best_model.joblib')

preds=best_model.predict(X_test)

best_model.init_layers(X_test.shape[0])
predictions=best_model.forward_pass(X_test)
print(predictions.shape)
predictions_loss=best_model.multi_label_loss(Y_test,predictions)

print("Hamming Accuracy: ", 1-hamming_loss(Y_test, preds))
print("Hamming Loss: ", hamming_loss(Y_test, preds))
print("Cross Entropy Loss: ", predictions_loss)
print("Precision: ", precision_score(Y_test, preds, average='micro',zero_division=0))
print("Recall: ", recall_score(Y_test, preds, average='micro',zero_division=0))
print("F1 Score: ", f1_score(Y_test, preds, average='micro',zero_division=0))
