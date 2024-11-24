import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.preprocessing import StandardScaler

module_path_mlp_classifier = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp'))
sys.path.append(module_path_mlp_classifier)
from mlp_classifier import MLP_Classifier

spotify_data = pd.read_csv("../../data/interim/spotify_modified.csv")

print(spotify_data.columns)

columns=['energy', 'tempo', 'danceability', 'valence','loudness', 'liveness','speechiness', 'acousticness','key', 'mode','time_signature', 'instrumentalness','popularity', 'explicit','duration_sec','track_genre']

columns1=['energy', 'tempo', 'danceability', 'valence','loudness', 'liveness','speechiness', 'acousticness','key', 'mode','time_signature', 'instrumentalness','popularity', 'explicit','duration_sec']

spotify_req=spotify_data[columns].copy()
spotify_req['explicit'] = spotify_req['explicit'].replace({True: 1, False: -1})
spotify_req.drop_duplicates(subset=columns1,inplace=True)

genre_mapping = {genre: idx for idx, genre in enumerate(spotify_req['track_genre'].unique())}
spotify_req['track_genre'] = spotify_req['track_genre'].map(genre_mapping)

# ---------------------------
# standardizing

features = spotify_req.drop('track_genre', axis=1)  # Drop 'track_genre' column to get features
target = spotify_req['track_genre']  # Keep 'track_genre' column separately
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)
spotify_scaled = pd.concat([scaled_features_df, target.reset_index(drop=True)], axis=1)

# ---------------------------

X=spotify_scaled.iloc[:, :-1].values
Y=spotify_scaled.iloc[:, -1].values

np.random.seed(42)

indices = np.random.permutation(len(X))
train_split = int(len(X) * 0.8)

X_train = X[indices[:train_split]]
Y_train = Y[indices[:train_split]].reshape(-1,1)
print(len(np.unique(Y_train)))

X_val = X[indices[train_split:]]
Y_val = Y[indices[train_split:]].reshape(-1,1)
print(len(np.unique(Y_val)))

mlp=MLP_Classifier(lr=0.001,optimizer='sgd',batch_size=256,num_hidden=2,num_neurons=[64,256],act_fun='tanh',num_epochs=500)
mlp.fit(X_train, Y_train, X_val, Y_val)


n_samples=X_train.shape[0]
num_classes=len(np.unique(Y_train))
best_accuracy=0
patience_counter=0
best_weights=None
Y_val_onehot=np.squeeze(np.eye(num_classes)[Y_val.astype(int).reshape(-1)])
Y_train_onehot=np.squeeze(np.eye(num_classes)[Y_train.astype(int).reshape(-1)])

if mlp.optimizer=='gd':
    batch_size=n_samples
elif mlp.optimizer=='sgd':
    batch_size=1
elif mlp.optimizer=='mini-batch':
    batch_size=mlp.batch_size

for epoch in range(mlp.num_epochs):
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
    y_val_acc_pred=y_val_acc_pred.reshape(-1,1)
    val_acc = accuracy_score(Y_val, y_val_acc_pred)

    mlp.init_layers(X_train.shape[0])
    y_train_pred=mlp.forward_pass(X_train)
    train_loss=mlp.loss(y_train_pred,Y_train_onehot)

    mlp.init_layers(X_train.shape[0])
    y_train_acc_pred=mlp.predict(X_train)
    y_train_acc_pred=y_train_acc_pred.reshape(-1,1)
    train_acc=accuracy_score(Y_train,y_train_acc_pred)

    print(f"Epoch {epoch+1}/{mlp.num_epochs}, Validation Loss: {val_loss}, Validation Accuracy: {val_acc*100:.2f}%, Training Loss: {train_loss}, Training Accuracy: {train_acc*100:.2f}%")
    
    if val_acc > best_accuracy:
        best_accuracy = val_acc
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

mlp.init_layers(X_val.shape[0])
Y_val_pred = mlp.predict(X_val)
Y_val_pred=Y_val_pred.reshape(-1,1)
accuracy=accuracy_score(Y_val,Y_val_pred)
precision_macro=precision_score(Y_val,Y_val_pred,average='macro',zero_division=0)
recall_macro=recall_score(Y_val,Y_val_pred,average='macro',zero_division=0)
f1_macro=f1_score(Y_val,Y_val_pred,average='macro')
precision_micro=precision_score(Y_val,Y_val_pred,average='micro',zero_division=0)
recall_micro=recall_score(Y_val,Y_val_pred,average='micro',zero_division=0)
f1_micro=f1_score(Y_val,Y_val_pred,average='micro')

print("Accuracy: ", accuracy)
print("Macro Precision: ", precision_macro)
print("Micro Precision: ", precision_micro)
print("Macro Recall: ", recall_macro)
print("Micro Recall: ", recall_micro)
print("Macro F1: ", f1_macro)
print("Micro F1: ", f1_micro)