# preprocessing the data
# Any code generated using LLMs is denoted by a comment of the sort - LLM ***

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df_wine = pd.read_csv('../../data/external/WineQT.csv')
# print(df_wine.head())
print(df_wine.shape)
df_wine=df_wine.dropna() # droping rows with missing values
print(df_wine.shape) # we see that zero rows are dropped
print(df_wine['Id'].nunique()) # we see that the column 'Id' is unique for each row and that there are no duplicates

# In total we have 12 attributes. The 'quality' attribute is the target variable. Lets describe the dataset for each of these attributes and summarize them into a table. 

# 1. Describe the dataset using mean, standard deviation, min, and max values for all attributes. (1)

description_wineqt = df_wine.describe().T[['mean', 'std', 'min', 'max']] # LLM ***
print(description_wineqt)

# 2. Draw a graph that shows the distribution of the various labels across the entire dataset. You are allowed to use standard libraries like Matplotlib. (2)

label_count=df_wine['quality'].value_counts()
plt.figure(figsize=(10, 6))
bars=label_count.plot(kind='bar', color='red')
plt.title('Distribution of Labels in WineQT dataset', fontsize=16)
plt.xlabel('Labels', fontsize=14)
plt.ylabel('Frequency', fontsize=14)
plt.xticks(rotation=0)
# LLM ***
for bar in bars.patches:
    bars.annotate(f'{bar.get_height()}', 
                  (bar.get_x() + bar.get_width() / 2, bar.get_height()), 
                  ha='center', va='bottom', fontsize=12)
# LLM ***
plt.tight_layout()
plt.savefig("figures/wineqt_label_distribution.png")

# 3. Normalise and standarize the data. Make sure to handle the missing or inconsistent data values if necessary. You can use sklearn for this.

missing_values = df_wine.isnull().sum()
print("Missing Values:\n", missing_values) # no missing values in the dataset
print(df_wine.dtypes) 
duplicates = df_wine.duplicated().sum()
print("Number of duplicates:", duplicates) # no duplicates in the dataset

# established that no records of missing values or inconsistencies in the dataset
min_max=MinMaxScaler()
standard=StandardScaler()
columns_to_scale = df_wine.columns[0:-2] # all columns except 'Id' and 'quality'
print(columns_to_scale)

df_wine[columns_to_scale] = standard.fit_transform(df_wine[columns_to_scale])
print(df_wine.describe().T[['mean', 'std', 'min', 'max']])

# LLM ***
df_wine = df_wine.drop(columns=['Id']) # drop the 'Id' column

# Adjust 'quality' values to start from 0
df_wine['quality'] = df_wine['quality'] - df_wine['quality'].min()

df_wine.to_csv('../../data/interim/modified_Wine_QT.csv', index=False)