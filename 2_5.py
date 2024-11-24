# 2.5 Analyzing Hyperparameters Effects (15 marks)
import os
import sys
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt

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

# 1. Effect of Non-linearity: Select four activation functions and vary only these functions while keeping other hyperparameters constant. Plot the loss versus the number of epochs for each activation function on a single graph. (5)

activation_functions = ['relu', 'tanh', 'sigmoid', 'linear']
lr=0.01
optimizer='sgd'
num_hidden=2
num_neurons=[16,8]
epochs=100

val_loss_arr = {act_fun: [] for act_fun in activation_functions}

for act_fun in activation_functions:
    print(f"Training for Activation Function: {act_fun}")
    mlp_activations=MLP_Classifier(lr=lr, act_fun=act_fun, optimizer=optimizer, num_hidden=num_hidden, num_neurons=num_neurons, num_epochs=epochs)
    val_loss = mlp_activations.fit_train(X_train, Y_train, X_val, Y_val)  
    val_loss_arr[act_fun] = val_loss

print(val_loss_arr['linear'])

plt.figure(figsize=(10, 6))
for act_fun in activation_functions:
    plt.plot(val_loss_arr[act_fun], label=act_fun)

plt.title('Validation Loss vs Epochs for Different Activation Functions', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(title='Activation Functions')
plt.grid()
plt.savefig("figures/activation_functions_val_loss.png")

# 2. Effect of Learning Rate: Choose four learning rates and vary only these rates while keeping other hyperparameters constant. Plot the loss versus the number of epochs for each learning rate on a single graph. (5)

activation_function = 'tanh'
lr=[0.001,0.01,0.05,0.1]
optimizer='sgd'
num_hidden=2
num_neurons=[16,8]
epochs=100

val_loss_arr = {l: [] for l in lr}

for l in lr:
    print(f"Training for learning rate: {l}")
    mlp_activations=MLP_Classifier(lr=l, act_fun=activation_function, optimizer=optimizer, num_hidden=num_hidden, num_neurons=num_neurons, num_epochs=epochs)
    val_loss = mlp_activations.fit_train(X_train, Y_train, X_val, Y_val)  
    val_loss_arr[l] = val_loss

plt.figure(figsize=(10, 6))
for l in lr:
    plt.plot(val_loss_arr[l], label=l)

plt.title('Validation Loss vs Epochs for Different Learning Rates', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(title='Learning Rates')
plt.grid()
plt.savefig("figures/learning_rates_val_loss.png")

# 3. Effect of Batch Size: Select four batch sizes and vary only these sizes while keeping other hyperparameters constant and the optimiser as mini- batch. Plot the loss versus the number of epochs for each batch size on a single graph. (5)

activation_function = 'tanh'
lr=0.01
optimizer='mini-batch'
num_hidden=2
num_neurons=[16,8]
epochs=100
batch_sizes=[16,32,64,128]

val_loss_arr = {b: [] for b in batch_sizes}

for b in batch_sizes:
    print(f"Training for batch size: {b}")
    mlp_activations=MLP_Classifier(lr=lr, act_fun=activation_function, optimizer=optimizer, num_hidden=num_hidden, num_neurons=num_neurons, num_epochs=epochs,batch_size=b)
    val_loss = mlp_activations.fit_train(X_train, Y_train, X_val, Y_val)  
    val_loss_arr[b] = val_loss
    preds=mlp_activations.predict(X_val)

plt.figure(figsize=(10, 6))
for b in batch_sizes:
    plt.plot(val_loss_arr[b], label=b)

plt.title('Validation Loss vs Epochs for Different Batch sizes', fontsize=16)
plt.xlabel('Epochs')
plt.ylabel('Validation Loss')
plt.legend(title='Learning Rates')
plt.grid()
plt.savefig("figures/batch_sizes_val_loss.png")