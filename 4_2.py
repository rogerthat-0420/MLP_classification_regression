# Train the autoencoder (10 marks)
# Now, using the dataset from assignment 1, train the auto encoder using the forward and backward pass methods of the class.

import os
import sys
import numpy as np
import time
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

module_path_autoencoder = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/autoencoder'))
sys.path.append(module_path_autoencoder)
from autoencoder import AutoEncoder

module_path_mlp_regression = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp'))
sys.path.append(module_path_mlp_regression)
from mlp_regression import MLP_Regression

module_path_knn=os.path.abspath(os.path.join(os.path.dirname(__file__),'../../models/knn'))
sys.path.append(module_path_knn)
from knn import KNN_Model

spotify_data = pd.read_csv("../../data/interim/spotify_modified.csv")

columns=['energy', 'tempo_normalized', 'danceability', 'valence','loudness_normalized', 'liveness','speechiness', 'acousticness','key', 'mode','time_signature', 'instrumentalness','popularity_normalized', 'explicit','duration_sec','track_genre']

columns1=['energy', 'tempo_normalized', 'danceability', 'valence','loudness_normalized', 'liveness','speechiness', 'acousticness','key', 'mode','time_signature', 'instrumentalness','popularity_normalized', 'explicit','duration_sec']

spotify_req=spotify_data[columns].copy()
spotify_req['explicit'] = spotify_req['explicit'].replace({True: 1, False: -1})
spotify_req.drop_duplicates(subset=columns1,inplace=True)

genre_mapping = {genre: idx for idx, genre in enumerate(spotify_req['track_genre'].unique())}
spotify_req['track_genre'] = spotify_req['track_genre'].map(genre_mapping)

# ---------------------------
features = spotify_req.drop('track_genre', axis=1)  # Drop 'track_genre' column to get features
target = spotify_req['track_genre']  # Keep 'track_genre' column separately

# Step 2: Standardize the features
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Convert the scaled features back to a DataFrame
scaled_features_df = pd.DataFrame(scaled_features, columns=features.columns)

# Step 4: Concatenate the scaled features with the target column
spotify_scaled = pd.concat([scaled_features_df, target.reset_index(drop=True)], axis=1)
# ---------------------------

X=spotify_scaled.iloc[:, :-1].values
Y=spotify_scaled.iloc[:, -1]

np.random.seed(42)
train_split = int(len(X) * 0.8)

X_train = X[:train_split]
X_val = X[train_split:]

optimal_dimension=12 # From PCA

spotify_autoencoder=AutoEncoder(lr=0.001,batch_size=16,latent_dim=optimal_dimension,num_hidden=3,num_neurons=[14,14],num_epochs=1000,optimizer='mini-batch')

spotify_autoencoder.fit(X_train,X_val)

joblib.dump(spotify_autoencoder, '../../models/autoencoder/4_2_autoencoder.joblib')

# 4.3 AutoEncoder + KNN (15 marks)

# 1.Use the trained model on the dataset to obtain the output of the encoder which will be the reduced dataset. Apply the KNN model from assignment 1 on the reduced dataset and return the validation F1 score, accuracy, precision, and recall. (5)

# reduced_x = spotify_autoencoder.get_latent(X)
# print(reduced_x.shape)
# Y=Y.values.reshape(-1,1)

# reduced_x_df = pd.DataFrame(reduced_x, columns=[f'latent_{i+1}' for i in range(reduced_x.shape[1])])
# Y_df=pd.DataFrame(Y,columns=['track_genre'])
# reduced_df = pd.concat([reduced_x_df, Y_df], axis=1)

# total_samples = reduced_df.shape[0]
# trainsize = int(0.8 * total_samples)

# train_reduced = reduced_df.iloc[:trainsize].to_numpy(dtype=np.float32)
# val_reduced = reduced_df.iloc[trainsize:]

# print(train_reduced.shape)

# k=15
# distance_metric='manhattan'
# print(f'Training dataset size={trainsize}')

# # training the knn model
# model_kdist = KNN_Model(k=k, dist_metric=distance_metric)
# model_kdist.fit(train_reduced)

# # predicting the labels for validation dataset
# predicted_labels = []
# starttime = time.time()
# print(f"Running model for k={k} and distance metric={distance_metric}")
# print(f"number of validation points={len(val_reduced)}")
# for i in range(len(val_reduced)):
#     print(f"Predicting for {i}th point")
#     pred = model_kdist.predict(test_point=val_reduced.iloc[i,:].to_numpy(dtype=np.float32).reshape(1,-1))
#     predicted_labels.append(pred)

# # metrics_perf = model_kdist.performance_measure(test_data=val_reduced, predicted_labels=predicted_labels)
# endtime = time.time()
# duration = (endtime - starttime) / 60
# print(f"Time taken for k={k} and distance metric={distance_metric}: {duration} minutes\n")

# accuracy=accuracy_score(val_reduced['track_genre'],predicted_labels)
# macro_precision=precision_score(val_reduced['track_genre'],predicted_labels,average='macro',zero_division=0)
# macro_recall=recall_score(val_reduced['track_genre'],predicted_labels,average='macro',zero_division=0)
# macro_f1=f1_score(val_reduced['track_genre'],predicted_labels,average='macro',zero_division=0)
# micro_precision=precision_score(val_reduced['track_genre'],predicted_labels,average='micro',zero_division=0)
# micro_recall=recall_score(val_reduced['track_genre'],predicted_labels,average='micro',zero_division=0)
# micro_f1=f1_score(val_reduced['track_genre'],predicted_labels,average='micro',zero_division=0)

# print(f"Accuracy: {accuracy}")
# print(f"Macro Precision: {macro_precision}")
# print(f"Macro Recall: {macro_recall}")
# print(f"Macro F1: {macro_f1}")
# print(f"Micro Precision: {micro_precision}")
# print(f"Micro Recall: {micro_recall}")
# print(f"Micro F1: {micro_f1}")