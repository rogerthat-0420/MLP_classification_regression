# Model Training & Hyperparameter Tuning using W&B
import os
import sys
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt
import wandb
import joblib
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

sweep_config = {
    'method': 'grid',
    'metric': {
        'name': 'Validation Accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'learning_rate': {
            'values': [0.01,0.05]
        },
        'num_epochs': {
            'values': [50, 100]
        },
        'neurons': {
            'values': [ [2,[8, 10]], [2,[16, 8]], [3,[8, 8, 4]], [3,[16, 8, 16]]]
        },
        'optimizers': {
            'values': ['sgd', 'gd', 'mini-batch']
        },
        'activation_functions': {
            'values': ['relu', 'tanh', 'sigmoid']
        }
    }
}

sweep_id = wandb.sweep(sweep_config, project="2.3 Model Training & Hyperparameter Tuning using W&B")

best_accuracy = 0
best_model=None
best_parameters={}
df_table=pd.DataFrame(columns=["Learning Rate", "Epochs","Optimizer","Neurons","Activation function", "Validation Accuracy", "Validation F1", "Validation Precision", "Validation Recall", "Validation Loss", "Training Accuracy", "Training F1", "Training Precision", "Training Recall", "Training Loss"])

def wandb_run():
    global best_accuracy, best_model, best_parameters, df_table
    with wandb.init() as run:

        if wandb.config.neurons[0]==2:
            run_name = f"lr_{wandb.config.learning_rate}_epochs_{wandb.config.num_epochs}_layers_{wandb.config.neurons[0]}_neurons_{wandb.config.neurons[1][0]}_{wandb.config.neurons[1][1]}_opt_{wandb.config.optimizers}_act_{wandb.config.activation_functions}"
        elif wandb.config.neurons[0]==3:
            run_name = f"lr_{wandb.config.learning_rate}_epochs_{wandb.config.num_epochs}_layers_{wandb.config.neurons[0]}_neurons_{wandb.config.neurons[1][0]}_{wandb.config.neurons[1][1]}_{wandb.config.neurons[1][2]}_opt_{wandb.config.optimizers}_act_{wandb.config.activation_functions}"
        wandb.run.name = run_name  

        lr=wandb.config.learning_rate
        optimizer = wandb.config.optimizers
        num_hidden_layers = wandb.config.neurons[0]
        num_neurons = wandb.config.neurons[1]
        activation_function = wandb.config.activation_functions
        epochs = wandb.config.num_epochs

        mlp = MLP_Classifier(lr=lr, act_fun=activation_function, optimizer=optimizer, num_hidden=num_hidden_layers, num_neurons=num_neurons,
                             num_epochs=epochs)
        mlp.fit(X_train, Y_train, X_val, Y_val)
        

        n_samples=X_train.shape[0]
        num_classes=len(np.unique(Y_train))
        best_loss=float('inf')
        patience_counter=0
        best_weights=None
        Y_val_onehot=np.squeeze(np.eye(num_classes)[Y_val.astype(int).reshape(-1)])
        Y_train_onehot=np.squeeze(np.eye(num_classes)[Y_train.astype(int).reshape(-1)])

        if optimizer=='gd':
            batch_size=n_samples
        elif optimizer=='sgd':
            batch_size=1
        elif optimizer=='mini-batch':
            batch_size=mlp.batch_size

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            X_train_shuffled = X_train[indices]
            Y_train_shuffled = Y_train_onehot[indices]

            for i in range(0, n_samples, batch_size):
                batch_end = min(i + batch_size, n_samples)
                X_batch = X_train_shuffled[i:batch_end]
                Y_batch = Y_train_shuffled[i:batch_end]

                mlp.init_layers(X_batch.shape[0])
                y_pred = mlp.forward_pass(X_batch)
                grad_w,grad_b=mlp.backward_pass(Y_batch,batch_size)
                mlp.update_params(grad_w,grad_b)

            # self.check_gradients(X_train_shuffled,Y_train_shuffled)

            
            # if self.type_class=='binary':
            #     current_loss = self.binary_loss(y_pred, Y_batch)
            # else:
            #     current_loss = self.loss(y_pred, Y_batch)

            # print(f'Epoch {epoch+1}/{self.num_epochs}, Loss: {current_loss}')
            
            mlp.init_layers(X_val.shape[0])
            y_val_pred = mlp.forward_pass(X_val)
            val_loss = mlp.loss(y_val_pred, Y_val_onehot)
            
            mlp.init_layers(X_val.shape[0])
            y_val_acc_pred = mlp.predict(X_val)
            val_acc = accuracy_score(Y_val, y_val_acc_pred)

            mlp.init_layers(X_train.shape[0])
            y_train_pred=mlp.forward_pass(X_train)
            train_loss=mlp.loss(y_train_pred,Y_train_onehot)
            y_train_acc_pred=mlp.predict(X_train)
            train_acc=accuracy_score(Y_train,y_train_acc_pred)

            wandb.log({"Validation Accuracy": val_acc*100, "Validation Loss": val_loss, "Training Accuracy": train_acc*100, "Training Loss": train_loss})
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = (mlp.weights.copy(), mlp.biases.copy())
                patience_counter = 0
            
            else:
                patience_counter+=1
                # print(f'Patience counter: {patience_counter}/{patience}')
            
            if patience_counter >= 5:
                if best_weights is not None:
                    mlp.weights, mlp.biases = best_weights
                #     print(f"Restoring best weights from epoch {epoch+1-patience}")
                # print(f"Early stopping at epoch {epoch+1}")
                break

        Y_val_pred = mlp.predict(X_val)
        Y_val_pred_loss=mlp.forward_pass(X_val)
        Y_val_true_loss=Y_val_onehot

        val_loss=mlp.loss(Y_val_pred_loss,Y_val_true_loss)
        f1_val=f1_score(Y_val, Y_val_pred, average='macro')
        precision_val=precision_score(Y_val, Y_val_pred, average='macro',zero_division=0)
        recall_val=recall_score(Y_val, Y_val_pred, average='macro',zero_division=0)
        accuracy_val=accuracy_score(Y_val, Y_val_pred)

        Y_train_pred = mlp.predict(X_train)
        Y_train_pred_loss=mlp.forward_pass(X_train)
        Y_train_true_loss=Y_train_onehot

        train_loss=mlp.loss(Y_train_pred_loss,Y_train_true_loss)
        f1_train=f1_score(Y_train, Y_train_pred, average='macro')
        precision_train=precision_score(Y_train, Y_train_pred, average='macro',zero_division=0)
        recall_train=recall_score(Y_train, Y_train_pred, average='macro',zero_division=0)
        accuracy_train=accuracy_score(Y_train, Y_train_pred)

        wandb_table_numeric = wandb.Table(columns=["Learning Rate", "Epochs","Optimizer","Neurons","Activation function", "Validation Accuracy", "Validation F1", "Validation Precision", "Validation Recall", "Validation Loss", "Training Accuracy", "Training F1", "Training Precision", "Training Recall", "Training Loss"])

        wandb_table_numeric.add_data(lr, epochs,optimizer,num_neurons,activation_function, accuracy_val*100, f1_val, precision_val, recall_val, val_loss, accuracy_train*100, f1_train, precision_train, recall_train, train_loss)

        wandb.log({"table":wandb_table_numeric})

        new_row = pd.DataFrame([{"Learning Rate": lr,
                         "Epochs": epochs,
                         "Optimizer": optimizer,
                         "Neurons": num_neurons,
                         "Activation function": activation_function,
                         "Validation Accuracy": accuracy_val,
                         "Validation F1": f1_val,
                         "Validation Precision": precision_val,
                         "Validation Recall": recall_val,
                         "Validation Loss": val_loss,
                         "Training Accuracy": accuracy_train,
                         "Training F1": f1_train,
                         "Training Precision": precision_train,
                         "Training Recall": recall_train,
                         "Training Loss": train_loss}])

        # Concatenate the new row with the existing DataFrame
        df_table = pd.concat([df_table, new_row], ignore_index=True)

        if accuracy_val > best_accuracy:
            best_accuracy = accuracy_val
            best_model = mlp  
            best_parameters = {
                "Learning Rate": lr,
                "Optimizer": optimizer,
                "Hidden Layers": num_hidden_layers,
                "Neurons": num_neurons,
                "Activation Function": activation_function,
                "Epochs": epochs
            }
            joblib.dump(best_model, "../../models/mlp/2_3_mlp_classification_wineQT_best_model.joblib")


wandb.agent(sweep_id, function = wandb_run)

print("Best Accuracy: ", best_accuracy)
print("Best Parameters: ", best_parameters)
# Plotting validation and training accuracy
df_table.to_csv("wine_mlp_classification_results.csv",index=False)