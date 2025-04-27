# brewgenie_prepare_data.py

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
import os

# Dataset paths
cocktail_path = 'datasets/hotaling_cocktails - Cocktails.csv'
beer_path = 'datasets/beer_data.csv'
spirits_path = 'datasets/spirits_data.csv'
wine_path = 'datasets/wine_data.csv'

# Load datasets
cocktails = pd.read_csv(cocktail_path)
beers = pd.read_csv(beer_path)
spirits = pd.read_csv(spirits_path)
wines = pd.read_csv(wine_path)

# Check columns
print("Cocktails Columns:", cocktails.columns.tolist())
print("Beers Columns:", beers.columns.tolist())
print("Spirits Columns:", spirits.columns.tolist())
print("Wines Columns:", wines.columns.tolist())

# Initialize embedder
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Updated generate_description function
def generate_description(row, type_):
    if type_ == 'cocktail':
        return f"{row['Cocktail Name']} - Ingredients: {row['Ingredients']}. Preparation: {row['Preparation']}. Garnish: {row['Garnish']}"
    elif type_ == 'beer':
        return f"{row['Name']} - Style: {row['Categories']}. ABV: {row['ABV']}"
    elif type_ == 'spirit':
        return f"{row['Name']} - Type: {row['Categories']}. Region: {row.get('Country', 'Unknown')}"
    elif type_ == 'wine':
        return f"{row['Name']} - Grape: {row['Categories']}. Region: {row.get('Country', 'Unknown')}"

# Generate descriptions and embeddings
cocktails['description'] = cocktails.apply(lambda x: generate_description(x, 'cocktail'), axis=1)
cocktails['embedding'] = cocktails['description'].apply(lambda x: embedder.encode(x).tolist())

beers['description'] = beers.apply(lambda x: generate_description(x, 'beer'), axis=1)
beers['embedding'] = beers['description'].apply(lambda x: embedder.encode(x).tolist())

spirits['description'] = spirits.apply(lambda x: generate_description(x, 'spirit'), axis=1)
spirits['embedding'] = spirits['description'].apply(lambda x: embedder.encode(x).tolist())

wines['description'] = wines.apply(lambda x: generate_description(x, 'wine'), axis=1)
wines['embedding'] = wines['description'].apply(lambda x: embedder.encode(x).tolist())

# Save embeddings
os.makedirs('datasets/embeddings', exist_ok=True)
cocktails.to_pickle('datasets/embeddings/cocktail_embeddings.pkl')
beers.to_pickle('datasets/embeddings/beer_embeddings.pkl')
spirits.to_pickle('datasets/embeddings/spirits_embeddings.pkl')
wines.to_pickle('datasets/embeddings/wine_embeddings.pkl')

print("✅ Embeddings generated and saved for cocktails, beers, spirits, and wines!")
