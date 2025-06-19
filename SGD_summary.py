
import os
#os.chdir(r"C:\Users\tom10\Desktop\statistics\מוסמך שנה א\כריית מידע- פרויקט")

#############################################

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier

df = pd.read_parquet(r'C:\Users\saarb\Desktop\courses\1MA\third_year\project_in_data_mining\ArxivCategoryPrediction\data\arxiv_data.parquet')#read_csv(r'C:\Users\saarb\Desktop\courses\1MA\third_year\project_in_data_mining\ArxivCategoryPrediction\data\df_all.csv')
summary_embeddings = np.load(r'C:\Users\saarb\Desktop\courses\1MA\third_year\project_in_data_mining\ArxivCategoryPrediction\data\summary_embeddings.npy')


df['primary_category'] = df['categories'].apply(lambda x: x.split(' ')[0])  # Take the first category if multiple are present
df['primary_category'] = df['primary_category'].apply(lambda x: x.split('.')[0])
df['primary_category'] = df['primary_category'].replace({i: 'ph' for i in ['astro-ph', 'cond-mat', 'gr-qc', 'hep-ex',
                                                                           'hep-lat', 'hep-ph', 'hep-th', 'nlin',
                                                                           'nucl-ex', 'nucl-th', 'physics', 'quant-ph',
                                                                           'math-ph', 'acc-phys', 'adap-org', 'ao-sci',
                                                                           'atom-ph', 'bayes-an', 'chao-dyn', 'chem-ph',
                                                                           'comp-gas', 'mtrl-th', 'patt-sol', 'plasm-ph',
                                                                           'solv-int', 'supr-con']})
df['primary_category'] = df['primary_category'].replace({i: 'math' for i in ['alg-geom', 'dg-ga', 'q-alg']})
df['primary_category'] = df['primary_category'].replace({i: 'cs' for i in ['cmp-lg']})
df['primary_category'] = df['primary_category'].replace({i: 'q-fin' for i in ['funct-an']})



# Keep only categories with at least 2 samples
category_counts = df['primary_category'].value_counts()
valid_categories = category_counts[category_counts >= 2].index
valid_df = df['primary_category'].isin(valid_categories)

# Filter both the embeddings and categories
filtered_embeddings = summary_embeddings[valid_df]
filtered_categories = df['primary_category'][valid_df]

# Now encode and split
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(filtered_categories)
mapping_dict = {index: label for index, label in enumerate(label_encoder.classes_)}


x_train, x_test, y_train, y_test = train_test_split(
    filtered_embeddings,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SGD Classifier
clf = SGDClassifier(
    loss='log_loss',  # for logistic regression
    max_iter=1000,
    random_state=1234,
    n_jobs=-1,
    fit_intercept=False
)
losses=['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']
# Train model
clf.fit(x_train, y_train)

# predictions
y_pred = clf.predict(x_test)

# accuracy
print(clf.score(x_test, y_test))
conf_mat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
conf_mat.replace(mapping_dict, inplace=True)


##### ConfusionMatrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Calculate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a confusion matrix display
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=mapping_dict.keys())  # Since we made it binary


ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
fig, ax = plt.subplots(figsize=(10,10))
plt.show()
# Plot the confusion matrix
plt.figure(figsize=(100, 100))
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_cm(y_true, y_pred, figsize=(10, 10)):
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
    cm_sum = np.sum(cm, axis=1, keepdims=True)
    cm_perc = cm / cm_sum.astype(float) * 100
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_perc[i, j]
            if i == j:
                s = cm_sum[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p, c)
    cm = pd.DataFrame(cm, index=np.unique(y_true), columns=np.unique(y_true))
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, cmap="YlGnBu", annot=annot, fmt='', ax=ax)


plot_cm(y_test, y_pred)