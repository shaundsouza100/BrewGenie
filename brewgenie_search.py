import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from dotenv import load_dotenv
import os
from openai import OpenAI

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
openai = OpenAI(api_key=api_key)

# Define beverage datasets and embeddings
datasets = {
    "cocktail": {
        "data": pd.read_pickle('datasets/embeddings/cocktail_embeddings.pkl'),
        "embedder": SentenceTransformer('paraphrase-MiniLM-L6-v2')
    },
    "beer": {
        "data": pd.read_pickle('datasets/embeddings/beer_embeddings.pkl'),
        "embedder": SentenceTransformer('paraphrase-MiniLM-L6-v2')
    },
    "spirit": {
        "data": pd.read_pickle('datasets/embeddings/spirits_embeddings.pkl'),
        "embedder": SentenceTransformer('paraphrase-MiniLM-L6-v2')
    },
    "wine": {
        "data": pd.read_pickle('datasets/embeddings/wine_embeddings.pkl'),
        "embedder": SentenceTransformer('paraphrase-MiniLM-L6-v2')
    }
}

# Beverage type detection based on query
def detect_beverage_type(query):
    query_lower = query.lower()
    if "cocktail" in query_lower:
        return "cocktail"
    elif "beer" in query_lower:
        return "beer"
    elif "wine" in query_lower:
        return "wine"
    elif "spirit" in query_lower or "whiskey" in query_lower or "vodka" in query_lower:
        return "spirit"
    else:
        return "cocktail"  # Default to cocktail

# Search function
def search_beverages(user_query, top_n=3):
    beverage_type = detect_beverage_type(user_query)
    dataset_info = datasets[beverage_type]
    df = dataset_info["data"]
    embedder = dataset_info["embedder"]
    
    # Generate query embedding
    query_embedding = embedder.encode(user_query, convert_to_tensor=True)
    # Convert stored embeddings
    beverage_embeddings = df['embedding'].apply(lambda x: torch.tensor(x))
    # Compute similarities
    similarities = beverage_embeddings.apply(lambda emb: util.pytorch_cos_sim(query_embedding, emb).item())
    df['similarity'] = similarities
    top_matches = df.nlargest(top_n, 'similarity')
    return top_matches, beverage_type

# User interaction loop
if __name__ == "__main__":
    query = input("Describe your beverage preference: ")
    results, beverage_type = search_beverages(query)
    
    print(f"\nTop {beverage_type.title()} Recommendations:")
    for idx, row in results.iterrows():
        if beverage_type == 'cocktail':
            beverage_name = row['Cocktail Name']
        else:
            beverage_name = row['Name']
        print(f"\n🍹 {beverage_name} (Similarity: {round(row['similarity'], 2)})")
        print(f"Description: {row['description']}")

    
    # Follow-up for recipe
    follow_up = input("\nWould you like a recipe or more info on any option? (yes/no): ")
    if follow_up.lower() == 'yes':
        selected = input("Type the name of the beverage you're interested in: ")
        # Use GPT for detailed output
        messages = [{"role": "user", "content": f"Give me a detailed recipe or tasting notes for {selected} {beverage_type}."}]
        response = openai.chat.completions.create(model="gpt-3.5-turbo", messages=messages)
        print("\n🔍 GPT Response:")
        print(response.choices[0].message.content)
