# 2.7 Analysis
import os
import sys
import pandas as pd
import numpy as np   
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score, hamming_loss

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

best_model=joblib.load('../../models/mlp/2_6_mlp_classification_advertisement_best_model.joblib')

preds=best_model.predict(X)
print(len(preds))

# LLM ***
class_labels = data_adv.columns[-8:]

# Convert the predictions from 0/1 to the corresponding class names
predicted_classes = []
for pred in preds:
    active_classes = [class_labels[i] for i, value in enumerate(pred) if value == 1]
    predicted_classes.append(active_classes if active_classes else ['None'])

# Convert the true labels from 0/1 to the corresponding class names
true_classes = []
for true in Y:
    active_classes = [class_labels[i] for i, value in enumerate(true) if value == 1]
    true_classes.append(active_classes if active_classes else ['None'])

# Create a DataFrame to display the true and predicted multi-classes side by side
comparison_df = pd.DataFrame({'True Classes': true_classes, 'Predicted Classes': predicted_classes})

# Create a figure and axis to plot the table
fig, ax = plt.subplots(figsize=(12, 200))

# Hide the axes
ax.axis('off')
ax.axis('tight')

# Plot the table
table = ax.table(cellText=comparison_df.values, colLabels=comparison_df.columns, cellLoc='center', loc='center')

# Adjust the table properties
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(comparison_df.columns))))

# Save the table as an image
plt.savefig("figures/2_7_analysis.png", bbox_inches='tight', dpi=300)

# LLM ***