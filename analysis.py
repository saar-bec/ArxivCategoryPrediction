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
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

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




# accuracy

accuracy = accuracy_score(res['y_test'], res['y_pred'])
y_test = res["y_test"]
y_pred = res["y_pred"]


# precision and recall per category
mapping_dict = {
    0: "Computer Science",
    1: "Economics",
    2: "Electrical Engineering and Systems Science",
    3: "Mathematics",
    4: "Physics",
    5: "Quantitative Biology",
    6: "Quantitative Finance",
    7: "Statistics"
}

precision_per_cat = precision_score(y_test, y_pred, average=None)
recall_per_cat = recall_score(y_test, y_pred, average=None)

# Create a DataFrame with readable class names
class_names = [mapping_dict[i] for i in range(len(precision_per_cat))]

metrics_df = pd.DataFrame({
    'Category': class_names,
    'Precision': precision_per_cat,
    'Recall': recall_per_cat
})

metrics_df = metrics_df.round(4)

#f1 score
macro_f1 = f1_score(y_test, y_pred, average='macro')

micro_f1 = f1_score(y_test, y_pred, average='micro')

#baseline_accuracy
most_frequent_class = np.bincount(y_test).argmax() # most frequent category in y_test
baseline_predictions = np.full_like(y_test, fill_value=most_frequent_class) # prediction array where every prediction is the most frequent class
baseline_accuracy = accuracy_score(y_test, baseline_predictions)



