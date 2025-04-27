#brewgenie_embeddings

import pandas as pd
from sentence_transformers import SentenceTransformer

# Load your cocktail dataset
df = pd.read_csv('hotaling_cocktails - Cocktails.csv')

# Keep only Cocktail Name, Ingredients, and Preparation
df_clean = df[['Cocktail Name', 'Ingredients', 'Preparation']].dropna(subset=['Cocktail Name', 'Ingredients'])
df_clean['Preparation'] = df_clean['Preparation'].fillna('Preparation details not available.')

# Initialize the Sentence Transformer model
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Combine Cocktail Name + Ingredients for embeddings
cocktail_texts = df_clean.apply(lambda row: f"{row['Cocktail Name']}: {row['Ingredients']}", axis=1)

# Generate embeddings
df_clean['embedding'] = embedder.encode(cocktail_texts.tolist()).tolist()

# Save the cleaned dataset with embeddings
df_clean.to_pickle('cocktail_embeddings.pkl')

print("✅ Embeddings generated and saved as 'cocktail_embeddings.pkl'!")
