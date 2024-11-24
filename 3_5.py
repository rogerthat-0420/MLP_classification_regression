# Mean Squared Error vs Binary Cross Entropy

# In this part, we will perform binary classification on the Pima Indians Diabetes dataset (diabetes.csv), using MSE and BCE loss function in the final layer and observe their effects.

import pandas as pd
import numpy as np
import joblib
import os
import sys
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

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

np.random.seed(42)
indices = np.random.permutation(X.shape[0])

train_split = int(0.8 * X.shape[0])

X_train = X[indices[:train_split]]
Y_train = Y[indices[:train_split]].reshape(-1,1)

X_val = X[indices[train_split:]]    
Y_val = Y[indices[train_split:]].reshape(-1,1)

val_test_split = int(len(X_val) * 0.8)

X_test = X_val[val_test_split:]
Y_test = Y_val[val_test_split:]

X_val = X_val[:val_test_split]
Y_val = Y_val[:val_test_split]


print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape,X_test.shape,Y_test.shape)

# mse model

mlp_mse=MLP_Regression(lr=0.001, num_epochs=100, num_hidden=0, num_neurons=[], act_fun='sigmoid',optimizer='sgd',logistic=True,batch_size=8,loss='mse')
mlp_mse.fit(X_train,Y_train,X_val,Y_val)

if mlp_mse.optimizer=='gd':
    batch_size=X_train.shape[0]
elif mlp_mse.optimizer=='mini-batch':
    batch_size=mlp_mse.batch_size
else:
    batch_size=1

print(mlp_mse.layer_sizes)

n_samples = X_train.shape[0]
best_loss=float('inf')
patience_counter=0
best_weights=None
validation_loss=[]

# training the model
for epoch in range(mlp_mse.num_epochs):
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]

    for i in range(0, n_samples, batch_size):
        batch_end=min(i+batch_size,n_samples)
        X_batch = X_train_shuffled[i:batch_end]
        Y_batch = Y_train_shuffled[i:batch_end]

        mlp_mse.init_layers(X_batch.shape[0])
        y_pred=mlp_mse.forward_pass(X_batch)
        grad_w,grad_b = mlp_mse.backward_pass(Y_batch,batch_size)
        mlp_mse.update_params(grad_w,grad_b)
        val_loss = mlp_mse.loss_func(y_pred,Y_batch)

    mlp_mse.init_layers(X_val.shape[0])
    y_val_pred=mlp_mse.forward_pass(X_val)
    val_loss = mlp_mse.loss_func(y_val_pred,Y_val)

    mlp_mse.init_layers(X_train.shape[0])
    y_train_pred=mlp_mse.forward_pass(X_train)
    train_loss = mlp_mse.loss_func(y_train_pred,Y_train)

    validation_loss.append(val_loss)

    print("Epoch: ",epoch," Train Loss: ",train_loss," Val Loss: ",val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        best_weights=mlp_mse.weights, mlp_mse.biases
        patience_counter=0
    else:
        patience_counter+=1

    if patience_counter>=10:
        if best_weights is not None:
            mlp_mse.weights,mlp_mse.biases=best_weights
            print("Early Stopping")
        break


y_val_pred=mlp_mse.forward_pass(X_val)
val_loss=mlp_mse.loss_func(y_val_pred,Y_val)
print("Final Validation Loss: ",val_loss)
y_val_acc=mlp_mse.predict(X_val)
print("Validation Accuracy: ",accuracy_score(Y_val,y_val_acc))

plt.plot(range(len(validation_loss)), validation_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs Number of Epochs for MSE loss SGD')
plt.savefig("figures/3_5_mse_loss_sgd.png")
plt.close()

joblib.dump(mlp_mse, '../../models/mlp/3_5_logistic_mse_sgd.joblib')

# bce model
n_samples = X_train.shape[0]
best_loss=float('inf')
patience_counter=0
best_weights=None
validation_loss=[]

mlp_bce=MLP_Regression(lr=0.001, num_epochs=100, num_hidden=0, num_neurons=[], act_fun='sigmoid',optimizer='sgd',logistic=True,batch_size=8,loss='bce')
mlp_bce.fit(X_train,Y_train,X_val,Y_val)

if mlp_bce.optimizer=='gd':
    batch_size=X_train.shape[0]
elif mlp_bce.optimizer=='mini-batch':
    batch_size=mlp_bce.batch_size
else:
    batch_size=1

print(mlp_bce.layer_sizes)

n_samples = X_train.shape[0]
best_loss=float('inf')
patience_counter=0
best_weights=None
validation_loss=[]

# training the model
for epoch in range(mlp_bce.num_epochs):
    indices = np.random.permutation(X_train.shape[0])
    X_train_shuffled = X_train[indices]
    Y_train_shuffled = Y_train[indices]

    for i in range(0, n_samples, batch_size):
        batch_end=min(i+batch_size,n_samples)
        X_batch = X_train_shuffled[i:batch_end]
        Y_batch = Y_train_shuffled[i:batch_end]

        mlp_bce.init_layers(X_batch.shape[0])
        y_pred=mlp_bce.forward_pass(X_batch)
        grad_w,grad_b = mlp_bce.backward_pass(Y_batch,batch_size)
        mlp_bce.update_params(grad_w,grad_b)
        val_loss = mlp_bce.loss_func(y_pred,Y_batch)

    mlp_bce.init_layers(X_val.shape[0])
    y_val_pred=mlp_bce.forward_pass(X_val)
    val_loss = mlp_bce.loss_func(y_val_pred,Y_val)

    mlp_bce.init_layers(X_train.shape[0])
    y_train_pred=mlp_bce.forward_pass(X_train)
    train_loss = mlp_bce.loss_func(y_train_pred,Y_train)

    validation_loss.append(val_loss)

    print("Epoch: ",epoch," Train Loss: ",train_loss," Val Loss: ",val_loss)

    if val_loss < best_loss:
        best_loss = val_loss
        best_weights=mlp_bce.weights, mlp_bce.biases
        patience_counter=0
    else:
        patience_counter+=1

    if patience_counter>=5:
        if best_weights is not None:
            mlp_bce.weights,mlp_bce.biases=best_weights
            print("Early Stopping")
        break


y_val_pred=mlp_bce.forward_pass(X_val)
val_loss=mlp_bce.loss_func(y_val_pred,Y_val)
print("Final Validation Loss: ",val_loss)
y_val_acc=mlp_bce.predict(X_val)
print("Validation Accuracy: ",accuracy_score(Y_val,y_val_acc))

plt.plot(range(len(validation_loss)), validation_loss)
plt.xlabel('Number of Epochs')
plt.ylabel('Validation Loss')
plt.title('Validation Loss vs Number of Epochs for BCE loss SGD')
plt.savefig("figures/3_5_bce_loss_sgd.png")
plt.close()

joblib.dump(mlp_bce, '../../models/mlp/3_5_logistic_bce_sgd.joblib')


