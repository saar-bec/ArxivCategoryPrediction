import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
import tensorflow_hub as hub
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


df = pd.read_csv(r'/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/arxiv_data/arxiv_data.csv')

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

try:
    combined_embeddings_df = pd.read_csv('/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/ArxivCategoryPrediction/data/combined_embeddings.csv')

except:
    # Load the Universal Sentence Encoder
    #module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
    module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
    #model = hub.module_v2(module_url)
    model = hub.load(module_url)
    print("module %s loaded" % module_url)

    def embed(strings):
        return model(strings)

    # Function to create embeddings for a column
    def create_embeddings_for_column(column_data):
        # Convert to list of strings and replace missing values
        texts = column_data.fillna('').astype(str).tolist()
        embeddings = embed(texts)
        return np.array(embeddings)

    # Create embeddings for each column
    title_embeddings = create_embeddings_for_column(df['title'])
    print("title")
    summary_embeddings1 = create_embeddings_for_column(df['abstract'].iloc[:450000,])
    summary_embeddings2 = create_embeddings_for_column(df['abstract'].iloc[450000:900000,])
    summary_embeddings3 = create_embeddings_for_column(df['abstract'].iloc[900000:1350000,])
    summary_embeddings4 = create_embeddings_for_column(df['abstract'].iloc[1350000:1800000,])
    summary_embeddings5 = create_embeddings_for_column(df['abstract'].iloc[1800000:2250000,])
    summary_embeddings6 = create_embeddings_for_column(df['abstract'].iloc[2250000:,])

    # Create column names for each embedding type
    title_cols = [f'title_emb_{i}' for i in range(title_embeddings.shape[1])]
    summary_cols = [f'summary_emb_{i}' for i in range(summary_embeddings1.shape[1])]

    # Create separate dataframes for each embedding type
    title_df = pd.DataFrame(title_embeddings, columns=title_cols)
    summary_df = pd.concat([
                            pd.DataFrame(summary_embeddings1, columns=summary_cols),
                            pd.DataFrame(summary_embeddings2, columns=summary_cols),
                            pd.DataFrame(summary_embeddings3, columns=summary_cols),
                            pd.DataFrame(summary_embeddings4, columns=summary_cols),
                            pd.DataFrame(summary_embeddings5, columns=summary_cols),
                            pd.DataFrame(summary_embeddings6, columns=summary_cols)],
                           ignore_index=True)

    # Combine all embeddings into one dataframe
    # title_df = pd.read_csv('/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/ArxivCategoryPrediction/data/title_embeddings.csv')
    # summary_df = pd.read_csv('/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/ArxivCategoryPrediction/data/abstract_embeddings.csv')
    # title_df.drop('Unnamed: 0', axis=1, inplace=True)
    # summary_df.drop('Unnamed: 0', axis=1, inplace=True)
    combined_embeddings_df = pd.concat([title_df, summary_df], axis=1)

    combined_embeddings_df.to_csv('/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/ArxivCategoryPrediction/data/combined_embeddings.csv')


# Now encode and split
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['primary_category'])
mapping_dict = {index: label for index, label in enumerate(label_encoder.classes_)}
del df
combined_embeddings_df = pd.read_csv('/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/ArxivCategoryPrediction/data/combined_embeddings.csv')

x_train, x_test, y_train, y_test = train_test_split(
    combined_embeddings_df,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# SGD Classifier
clf = SGDClassifier(
    loss='log_loss',  # for logistic regression
    n_jobs=-1,
    fit_intercept=False
)

# Train model
clf.fit(x_train, y_train)

# predictions
y_pred = clf.predict(x_test)
pd.DataFrame({'y_pred': y_pred, 'y_test':y_test}).to_csv('/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/ArxivCategoryPrediction/data/results.csv')
# accuracy
print(clf.score(x_test, y_test))
conf_mat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
conf_mat.replace(mapping_dict, inplace=True)


# accuracy
accuracy = accuracy_score(df['y_test'], df['y_pred'])
print(f"Accuracy: {accuracy}")


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

print(metrics_df)

#f1 score
macro_f1 = f1_score(y_test, y_pred, average='macro')
print("Macro F1 Score:", round(macro_f1, 4))

micro_f1 = f1_score(y_test, y_pred, average='micro')
print("Micro F1 Score:", round(micro_f1, 4))

#baseline_accuracy
most_frequent_class = np.bincount(y_test).argmax() # most frequent category in y_test
baseline_predictions = np.full_like(y_test, fill_value=most_frequent_class) # prediction array where every prediction is the most frequent class
baseline_accuracy = accuracy_score(y_test, baseline_predictions)
print("Baseline Accuracy (Most Frequent Class):", round(baseline_accuracy, 4))



