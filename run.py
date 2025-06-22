import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
import tensorflow_hub as hub

df = pd.read_parquet(r'C:\Users\saarb\Desktop\courses\1MA\third_year\project_in_data_mining\ArxivCategoryPrediction\arxiv_data\arxiv_data.parquet') # '/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/arxiv_data.parquet')
# df = df.iloc[0:100000]
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
    combined_embeddings_df = pd.read_parquet('/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/combined_embeddings.parquet')

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
    title_embeddings = create_embeddings_for_column(df['title']) # [embed([strings]) for strings in df['title']] #
    summary_embeddings = create_embeddings_for_column(df['abstract']) # 'summary' # [embed([strings]) for strings in df['abstract']] #
    authors_embeddings = create_embeddings_for_column(df['authors']) # [embed([strings]) for strings in df['authors']] #

    # Create column names for each embedding type
    title_cols = [f'title_emb_{i}' for i in range(title_embeddings.shape[1])]
    summary_cols = [f'summary_emb_{i}' for i in range(summary_embeddings.shape[1])]
    authors_cols = [f'authors_emb_{i}' for i in range(authors_embeddings.shape[1])]

    # Create separate dataframes for each embedding type
    title_df = pd.DataFrame(title_embeddings, columns=title_cols)
    summary_df = pd.DataFrame(summary_embeddings, columns=summary_cols)
    authors_df = pd.DataFrame(authors_embeddings, columns=authors_cols)

    # Combine all embeddings into one dataframe
    combined_embeddings_df = pd.concat([title_df, summary_df, authors_df], axis=1)

    combined_embeddings_df.to_parquet('/sci/labs/orzuk/orzuk/teaching/big_data_project_52017/Tom_Saar/combined_embeddings.parquet')


# Now encode and split
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['primary_category'])
mapping_dict = {index: label for index, label in enumerate(label_encoder.classes_)}


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
    max_iter=1000,
    random_state=1234,
    n_jobs=-1,
    fit_intercept=False
)
# losses=['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive']

# Train model
clf.fit(x_train, y_train)

# predictions
y_pred = clf.predict(x_test)

# accuracy
print(clf.score(x_test, y_test))
conf_mat = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
conf_mat.replace(mapping_dict, inplace=True)




