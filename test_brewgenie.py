# test_brewgenie.py

from brewgenie_rag_search import get_rag_response

# Predefined test cases (diverse queries)
test_queries = [
    "Suggest a fruity cocktail",
    "Recommend a wine from France",
    "What is a good whiskey for beginners?",
    "Give me a beer with low bitterness",
    "Tell me about a tropical rum drink"
]

# Optional: Define expected keywords for sanity checks
expected_keywords = {
    "cocktail": ["cocktail", "recipe", "ingredients"],
    "wine": ["wine", "grape", "region"],
    "whiskey": ["whiskey", "flavor", "aged"],
    "beer": ["beer", "lager", "ale", "bitterness"],
    "rum": ["rum", "tropical", "coconut"]
}

def run_tests():
    print("Running BrewGenie RAG Pipeline Tests...\n")

    for query in test_queries:
        print(f"📝 User Query: {query}")
        response = get_rag_response(query)
        print(f"🤖 BrewGenie Response: {response}\n")

        # Sanity check: Look for expected keywords
        category = query.split()[2]  # crude extraction (e.g., 'cocktail', 'wine')
        keywords = expected_keywords.get(category.lower(), [])
        if any(keyword.lower() in response.lower() for keyword in keywords):
            print("✅ Response looks relevant!\n")
        else:
            print("⚠️ Response may need review.\n")
        print("=" * 60)

if __name__ == "__main__":
    run_tests()
