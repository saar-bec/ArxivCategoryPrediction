
import os
os.chdir(r"C:\Users\tom10\Desktop\statistics\מוסמך שנה א\כריית מידע- פרויקט")


import pandas as pd
import numpy as np
import tensorflow as tf



import tensorflow_hub as hub
import shutil
import os



# Load the Universal Sentence Encoder
#module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
module_url = "https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2"
#model = hub.module_v2(module_url)
model = hub.load(module_url)
print("module %s loaded" % module_url)



import tensorflow_hub as hub
#embed = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/TensorFlow2/universal-sentence-encoder/2")


def embed(strings):
    return model(strings)

# Load your CSV file
#df = pd.read_csv('df_all.csv')
df = pd.read_parquet('arxiv_data.parquet')


# Function to create embeddings for a column
def create_embeddings_for_column(column_data):
    # Convert to list of strings and replace missing values
    texts = column_data.fillna('').astype(str).tolist()
    embeddings = embed(texts)
    return np.array(embeddings)

# Create embeddings for each column
title_embeddings = create_embeddings_for_column(df['title'])
summary_embeddings = create_embeddings_for_column(df['abstract'])#'summary'
authors_embeddings = create_embeddings_for_column(df['authors'])


#save embeddings
np.save('title_embeddings.npy', title_embeddings)
np.save('summary_embeddings.npy', summary_embeddings)
np.save('authors_embeddings.npy', authors_embeddings)

print(title_embeddings.shape)
print(title_embeddings[0][:10])

print(summary_embeddings.shape)
print(summary_embeddings[0][:10])

print(authors_embeddings.shape)
print(authors_embeddings[0][:10])







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

# Save the combined DataFrame
combined_embeddings_df.to_csv('combined_embeddings.csv', index=False)

print("Combined embeddings shape:", combined_embeddings_df.shape)


