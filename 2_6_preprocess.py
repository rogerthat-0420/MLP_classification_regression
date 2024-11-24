import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np

advertisement = pd.read_csv('../../data/external/advertisement.csv')
print(len(advertisement))
advertisement.dropna(inplace=True)
advertisement['labels']=advertisement['labels'].str.split()

education_mapping = {
    'High School': 1,
    'Bachelor': 2,
    'Master': 3,
    'PhD':4
}
advertisement['education']=advertisement['education'].map(education_mapping)
advertisement['married'] = advertisement['married'].apply(lambda x: 1 if x == True else 0)

gender= pd.get_dummies(advertisement['gender'], prefix='gender',dtype=float)
advertisement = pd.concat([advertisement,gender],axis=1)
occupation = pd.get_dummies(advertisement['occupation'],prefix='occupation',dtype=float)
advertisement = pd.concat([advertisement,occupation],axis=1)

multi_label_binarizer=MultiLabelBinarizer()
labels = multi_label_binarizer.fit_transform(advertisement['labels'])
labels_df=pd.DataFrame(labels,columns=multi_label_binarizer.classes_)
advertisement=pd.concat([advertisement,labels_df],axis=1)

advertisement.drop(['most bought item','city','labels','gender','occupation'],axis=1,inplace=True)
advertisement=advertisement.astype(float)
# print(advertisement.columns)
# numeric_cols = advertisement.select_dtypes(include=[np.number]).columns
# print(numeric_cols)

# # standardize the data
data = advertisement.iloc[:, :-8].values
data = (data - data.mean(axis=0)) / data.std(axis=0)
advertisement.iloc[:, :-8] = data
print(advertisement.dtypes)
advertisement.to_csv('../../data/interim/advertisement_modified.csv', index=False)
