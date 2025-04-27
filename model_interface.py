import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize the OpenAI client
client = OpenAI(api_key=api_key)

# Function to query OpenAI
def query_openai(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  # or "gpt-4" if you have access
        messages=[
            {"role": "system", "content": "You are a helpful assistant for alcohol recommendations."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500
    )
    return response.choices[0].message.content.strip()

# Example function to get drink recommendations
def get_drink_recommendation(category, taste):
    prompt = f"Suggest a {category} drink that is {taste}."
    return query_openai(prompt)

# Example test run
if __name__ == "__main__":
    drink = get_drink_recommendation("cocktail", "fruity")
    print(f"Recommended Drink: {drink}")
