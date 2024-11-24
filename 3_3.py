import os
import sys
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt
import wandb
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

print("X_train",X_train.shape)
print("Y_train",Y_train.shape)
print("X_val",X_val.shape)
print("Y_val",Y_val.shape)
print("X_test",X_test.shape)
print("Y_test",Y_test.shape)

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'Validation MSE',
        'goal': 'minimize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01,0.05,0.025]
        },
        'neurons': {
            'values': [ [1,[20]], [2,[13, 6]], [2,[16, 8]], [3,[16, 8, 16]], [3,[16, 32, 16]]]
        },
        'optimizers': {
            'values': ['sgd', 'gd', 'mini-batch']
        },
        'activation_functions': {
            'values': ['relu', 'tanh', 'sigmoid']
        }
    }
}

sweep_id=wandb.sweep(sweep_config, project="SMAI A3 3.3 Model Training & Hyperparameter Tuning using W&B")

model_best_loss = float('inf')
best_model = None
best_parameters={}
df_table=pd.DataFrame(columns=["Learning Rate","Optimizer","Hidden Layers","Neurons","Activation Function","Training MSE","Training RMSE","Training R2","Validation MSE","Validation RMSE","Validation R2"])

def wandb_run():
    global best_loss, best_model, best_parameters, df_table

    with wandb.init() as run:

        if wandb.config.neurons[0]==1:
            run_name = f"lr_{wandb.config.learning_rate}_layers_{wandb.config.neurons[0]}_neurons_{wandb.config.neurons[1][0]}_opt_{wandb.config.optimizers}_act_{wandb.config.activation_functions}"
        elif wandb.config.neurons[0]==2:
            run_name = f"lr_{wandb.config.learning_rate}_layers_{wandb.config.neurons[0]}_neurons_{wandb.config.neurons[1][0]}_{wandb.config.neurons[1][1]}_opt_{wandb.config.optimizers}_act_{wandb.config.activation_functions}"
        elif wandb.config.neurons[0]==3:
            run_name = f"lr_{wandb.config.learning_rate}_layers_{wandb.config.neurons[0]}_neurons_{wandb.config.neurons[1][0]}_{wandb.config.neurons[1][1]}_{wandb.config.neurons[1][2]}_opt_{wandb.config.optimizers}_act_{wandb.config.activation_functions}"
        wandb.run.name = run_name 

        lr=wandb.config.learning_rate
        optimizer = wandb.config.optimizers
        num_hidden_layers = wandb.config.neurons[0]
        num_neurons = wandb.config.neurons[1]
        activation_function = wandb.config.activation_functions
        epochs=100

        mlp_reg=MLP_Regression(lr=lr, act_fun=activation_function, optimizer=optimizer, num_hidden=num_hidden_layers, num_neurons=num_neurons)

        mlp_reg.fit(X_train, Y_train, X_val, Y_val)

        if optimizer=='sgd':
            batch_size=1
        elif optimizer=='gd':
            batch_size=len(X_train)
        elif optimizer=='mini-batch':
            batch_size=32
        
        n_samples = X_train.shape[0]
        best_loss = float('inf')
        patience_counter=0
        best_weights=None

        # training loop
        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            Y_train_shuffled = Y_train[indices]

            for i in range(0, n_samples, batch_size):
                batch_end=min(i+batch_size,n_samples)
                X_batch = X_train_shuffled[i:batch_end]
                Y_batch = Y_train_shuffled[i:batch_end]

                mlp_reg.init_layers(X_batch.shape[0])
                y_pred=mlp_reg.forward_pass(X_batch)
                grad_w,grad_b = mlp_reg.backward_pass(Y_batch,batch_size)
                mlp_reg.update_params(grad_w,grad_b)

            mlp_reg.init_layers(X_val.shape[0])
            y_val_pred=mlp_reg.forward_pass(X_val)
            val_loss = mlp_reg.mse_loss(y_val_pred,Y_val)

            mlp_reg.init_layers(X_train.shape[0])
            y_train_pred=mlp_reg.forward_pass(X_train)
            train_loss = mlp_reg.mse_loss(y_train_pred,Y_train)

            wandb.log({"Training MSE":train_loss,"Validation MSE":val_loss})

            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = mlp_reg.weights,mlp_reg.biases
                patience_counter=0
            else:
                patience_counter+=1

            if patience_counter>=5:
                if best_weights is not None:
                    mlp_reg.weights,mlp_reg.biases=best_weights
                break
        
        mlp_reg.init_layers(X_val.shape[0])
        y_val_pred=mlp_reg.forward_pass(X_val)
        val_mse = mlp_reg.mse_loss(y_val_pred,Y_val)
        val_rmse = mlp_reg.rmse_loss(y_val_pred,Y_val)
        val_r2 = mlp_reg.rsquared(y_val_pred,Y_val)

        mlp_reg.init_layers(X_train.shape[0])
        y_train_pred=mlp_reg.forward_pass(X_train)
        train_mse = mlp_reg.mse_loss(y_train_pred,Y_train)
        train_rmse = mlp_reg.rmse_loss(y_train_pred,Y_train)
        train_r2 = mlp_reg.rsquared(y_train_pred,Y_train)

        wandb_table_numeric=wandb.Table(columns=["Learning Rate","Optimizer","Hidden Layers","Neurons","Activation Function","Training MSE","Training RMSE","Training R2","Validation MSE","Validation RMSE","Validation R2"])
                                                 
        wandb_table_numeric.add_data(lr,optimizer,num_hidden_layers,num_neurons,activation_function,train_mse,train_rmse,train_r2,val_mse,val_rmse,val_r2)

        new_row = pd.DataFrame([{"Learning Rate": lr,
                         "Optimizer": optimizer,
                         "Hidden Layers": num_hidden_layers,
                         "Neurons": num_neurons,
                         "Activation Function": activation_function,
                         "Training MSE": train_mse,
                         "Training RMSE": train_rmse,
                         "Training R2": train_r2,
                         "Validation MSE": val_mse,
                         "Validation RMSE": val_rmse,
                         "Validation R2": val_r2}])

        # Concatenate the new row with the existing DataFrame
        df_table = pd.concat([df_table, new_row], ignore_index=True)
        wandb.log({"table":wandb_table_numeric})

        if val_mse < model_best_loss:
            print("YES")
            best_loss = val_mse
            best_model = mlp_reg  # Store the best model
            best_parameters = {
                "Learning Rate": lr,
                "Optimizer": optimizer,
                "Hidden Layers": num_hidden_layers,
                "Neurons": num_neurons,
                "Activation Function": activation_function,
                "Epochs": epochs
            }
        joblib.dump(best_model, "../../models/mlp/3_3_mlp_regression_housing_best_model.joblib")


wandb.agent(sweep_id, function = wandb_run)


print("Best Loss: ", best_loss)
print("Best Parameters: ", best_model.model_parameters())

df_table.to_csv("housing_mlp_regression_results.csv",index=False)
