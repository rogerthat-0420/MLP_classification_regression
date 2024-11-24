# Data Preprocessing 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data=pd.read_csv("../../data/external/HousingData.csv")
print(data.head())
print(len(data))

medians=data.median()
data.fillna(medians, inplace=True)

# 1. Describe the dataset using mean, standard deviation, min, and max values for all attributes. (1)

house_describe=pd.DataFrame(columns=data.columns)
house_describe.loc['mean']=data.mean(axis=0)
house_describe.loc['std']=data.std(axis=0)
house_describe.loc['min']=data.min(axis=0)
house_describe.loc['max']=data.max(axis=0)
print(house_describe)

data.hist(figsize=(20,10),edgecolor='black')
plt.suptitle("Histogram showing the distribution of various attributes in Housing Data",size=20)
plt.savefig("figures/housing_hist.png")

# normalize and standardize the data
scaler = StandardScaler()
data_standardized = scaler.fit_transform(data)

data_standardized_df = pd.DataFrame(data_standardized, columns=data.columns)

house_standardized_describe=pd.DataFrame(columns=data_standardized_df.columns)
house_standardized_describe.loc['mean']=data_standardized_df.mean(axis=0)
house_standardized_describe.loc['std']=data_standardized_df.std(axis=0)
house_standardized_describe.loc['min']=data_standardized_df.min(axis=0)
house_standardized_describe.loc['max']=data_standardized_df.max(axis=0)
print(house_standardized_describe)

# saving the modified dataframe in a csv file.
data_standardized_df.to_csv("../../data/interim/housing_modified.csv",index=False)

X=data_standardized_df.iloc[:, :-1].values
Y=data_standardized_df.iloc[:, -1].values

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