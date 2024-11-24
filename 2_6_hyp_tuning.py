import os
import sys
import pandas as pd
import numpy as np   
import matplotlib.pyplot as plt
import wandb
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

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

sweep_id=wandb.sweep(sweep_config, project="2.6 Model Training & Hyperparameter Tuning using W&B")

best_accuracy = 0
best_model=None
best_paramaters={}
df_table=pd.DataFrame(columns=["Learning Rate","Epochs","Optimizer","Hidden Layers","Neurons","Activation Function","Validation Accuracy","Validation Hamming Loss","Validation Loss","Validation F1","Validation Precision","Validation Recall","Training Accuracy","Training Hamming Loss","Training Loss","Training F1","Training Precision","Training Recall"])

def wandb_run():
    global best_accuracy, best_model, best_parameters, df_table

    with wandb.init() as run:

        if wandb.config.neurons[0]==2:
            run_name = f"lr_{wandb.config.learning_rate}_epochs_{wandb.config.num_epochs}_layers_{wandb.config.neurons[0]}_neurons_{wandb.config.neurons[1][0]}_{wandb.config.neurons[1][1]}_opt_{wandb.config.optimizers}_act_{wandb.config.activation_functions}"
        elif wandb.config.neurons[0]==3:
            run_name = f"lr_{wandb.config.learning_rate}_epochs_{wandb.config.num_epochs}_layers_{wandb.config.neurons[0]}_neurons_{wandb.config.neurons[1][0]}_{wandb.config.neurons[1][1]}_{wandb.config.neurons[1][2]}_opt_{wandb.config.optimizers}_act_{wandb.config.activation_functions}"
        wandb.run.name = run_name 

        lr=wandb.config.learning_rate
        optimizers = wandb.config.optimizers
        num_hidden_layers = wandb.config.neurons[0]
        num_neurons = wandb.config.neurons[1]
        activation_function = wandb.config.activation_functions
        epochs = wandb.config.num_epochs

        mlp_multi=MLP_Multi_Classifier(lr=lr, act_fun=activation_function, optimizer=optimizers, num_hidden=num_hidden_layers, num_neurons=num_neurons, num_epochs=epochs)
        mlp_multi.fit(X_train,Y_train,X_val,Y_val)

        n_samples=mlp_multi.X.shape[0]
        patience_counter=0
        best_loss=float('inf')
        best_weights=None

        if mlp_multi.optimizer =='gd':
            batch_size=n_samples
        elif mlp_multi.optimizer=='sgd':
            batch_size=1
        elif mlp_multi.optimizer=='mini-batch':
            batch_size=mlp_multi.batch_size
        
        for epoch in range(mlp_multi.num_epochs):
            indices=np.random.permutation(n_samples)
            X_train_shuffled=mlp_multi.X[indices]
            Y_train_shuffled=mlp_multi.Y[indices]

            for i in range(0,n_samples,batch_size):
                batch_end=min(i+batch_size,n_samples)
                X_batch=X_train_shuffled[i:batch_end]
                Y_batch=Y_train_shuffled[i:batch_end]

                mlp_multi.init_layers(X_batch.shape[0])
                Y_pred=mlp_multi.forward_pass(X_batch)
                grad_w,grad_b=mlp_multi.backward_pass(Y_batch,batch_size)
                mlp_multi.update_params(grad_w,grad_b)
            
            mlp_multi.init_layers(X_val.shape[0])
            y_val_pred=mlp_multi.forward_pass(X_val)
            val_loss=mlp_multi.multi_label_loss(Y_val,y_val_pred)
            hamming_val_accuracy=mlp_multi.hamming_accuracy(Y_val,y_val_pred)
            hamming_val_loss=1-hamming_val_accuracy

            mlp_multi.init_layers(X_train.shape[0])
            y_train_pred=mlp_multi.forward_pass(X_train)
            train_loss=mlp_multi.multi_label_loss(Y_train,y_train_pred)
            hamming_train_accuracy=mlp_multi.hamming_accuracy(Y_train,y_train_pred)
            hamming_train_loss=1-hamming_train_accuracy

            wandb.log({"Validation Loss": val_loss, "Validation Accuracy": hamming_val_accuracy, "Training Loss": train_loss, "Training Accuracy": hamming_train_accuracy, "Validation Hamming Loss": hamming_val_loss, "Training Hamming Loss": hamming_train_loss})

            if hamming_val_loss<best_loss:
                best_loss=hamming_val_loss
                best_weights=(mlp_multi.weights.copy(),mlp_multi.biases.copy())
                patience_counter=0
            
            else:
                patience_counter+=1
            
            if patience_counter>=5:
                if best_weights is not None:
                    mlp_multi.weights,mlp_multi.biases=best_weights
                #     print(f"Restoring best weights from epoch {epoch+1-patience_counter}")
                # print(f"Early stopping at epoch {epoch+1}")
                break

        mlp_multi.init_layers(X_val.shape[0])
        Y_val_pred_loss=mlp_multi.forward_pass(X_val)
        Y_val_pred=mlp_multi.predict(X_val)

        val_accuracy=mlp_multi.hamming_accuracy(Y_val,Y_val_pred_loss)
        val_hamming_loss=1-val_accuracy
        val_loss=mlp_multi.multi_label_loss(Y_val,Y_val_pred_loss)
        f1_val=f1_score(Y_val,Y_val_pred,average='macro')
        precision_val=precision_score(Y_val,Y_val_pred,average='macro',zero_division=0)
        recall_val=recall_score(Y_val,Y_val_pred,average='macro',zero_division=0)

        mlp_multi.init_layers(X_train.shape[0])
        Y_train_pred_loss=mlp_multi.forward_pass(X_train)
        Y_train_pred=mlp_multi.predict(X_train)

        train_accuracy=mlp_multi.hamming_accuracy(Y_train,Y_train_pred_loss)
        train_hamming_loss=1-train_accuracy
        train_loss=mlp_multi.multi_label_loss(Y_train,Y_train_pred_loss)
        f1_train=f1_score(Y_train,Y_train_pred,average='macro')
        precision_train=precision_score(Y_train,Y_train_pred,average='macro',zero_division=0)
        recall_train=recall_score(Y_train,Y_train_pred,average='macro',zero_division=0)

        wandb_table_numeric=wandb.Table(["Learning Rate","Epochs","Optimizer","Hidden Layers","Neurons","Activation Function","Validation Accuracy","Validation Hamming Loss","Validation Loss","Validation F1","Validation Precision","Validation Recall","Training Accuracy","Training Hamming Loss","Training Loss","Training F1","Training Precision","Training Recall"])

        wandb_table_numeric.add_data(lr,epochs,optimizers,num_hidden_layers,num_neurons,activation_function,val_accuracy,val_hamming_loss,val_loss,f1_val,precision_val,recall_val,train_accuracy,train_hamming_loss,train_loss,f1_train,precision_train,recall_train)

        new_row=pd.DataFrame([{"Learning Rate":lr,"Epochs":epochs,"Optimizer":optimizers,"Hidden Layers":num_hidden_layers,"Neurons":num_neurons,"Activation Function":activation_function,"Validation Accuracy":val_accuracy,"Validation Hamming Loss":val_hamming_loss,"Validation Loss":val_loss,"Validation F1":f1_val,"Validation Precision":precision_val,"Validation Recall":recall_val,"Training Accuracy":train_accuracy,"Training Hamming Loss":train_hamming_loss,"Training Loss":train_loss,"Training F1":f1_train,"Training Precision":precision_train,"Training Recall":recall_train}])

        df_table=pd.concat([df_table,new_row],ignore_index=True)

        wandb.log({"table":wandb_table_numeric})

        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_model = mlp_multi  # Store the best model
            best_parameters = {
                "Learning Rate": lr,
                "Optimizer": optimizers,
                "Hidden Layers": num_hidden_layers,
                "Neurons": num_neurons,
                "Activation Function": activation_function,
                "Epochs": epochs
            }
            joblib.dump(best_model, "../../models/mlp/2_6_mlp_classification_advertisement_best_model.joblib")

wandb.agent(sweep_id, function = wandb_run)


print("Best Accuracy: ", best_accuracy)
print("Best Parameters: ", best_parameters)

df_table.to_csv("hyptun_advertisement_multi_classification_results.csv",index=False)
