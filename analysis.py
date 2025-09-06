import pandas as pd


mapping_dict = {0: 'cs', 1: 'econ', 2: 'eess', 3: 'math', 4: 'ph', 5: 'q-bio', 6: 'q-fin', 7: 'stat'}
res = pd.read_csv(r'C:\Users\saarb\Desktop\courses\1MA\third_year\project_in_data_mining\ArxivCategoryPrediction\data\results.csv').drop('Unnamed: 0', axis=1)
res.replace(mapping_dict, inplace=True)
y_test_counts = res['y_test'].value_counts()

accuracy = (res['y_test']==res['y_pred']).sum()/res.shape[0]
confusion_mat = pd.crosstab(res['y_test'], res['y_pred'], rownames=['Actual'], colnames=['Predicted'], normalize='all') # accuracy
confusion_mat = pd.crosstab(res['y_test'], res['y_pred'], rownames=['Actual'], colnames=['Predicted'], normalize='index') # recall



import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Example transition matrix
transition_matrix = confusion_mat
cols = transition_matrix.columns
index = transition_matrix.index

# Create a heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(transition_matrix, annot=True, cmap='Blues', fmt=".2f",
            xticklabels=cols,
            yticklabels=index)
plt.title('Transition Matrix Heatmap')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()