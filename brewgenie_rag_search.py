from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load FAISS index
vectorstore = FAISS.load_local(
    "brewgenie_faiss_index",
    OpenAIEmbeddings(model="text-embedding-ada-002"),
    allow_dangerous_deserialization=True
)

# Initialize GPT model (system prompt manually handled)
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)

# Create RetrievalQA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Final function with Prompt Engineering (context, system prompt, error handling)
def get_rag_response(query, chat_history=None):
    # Edge Case: Handle empty input
    if not query.strip():
        return "🍹 Hey! Tell me what drink you're in the mood for!"

    # Build context from chat history (limit to last 6 messages)
    context = ""
    if chat_history:
        recent_history = chat_history[-6:]  # Use last 6 messages
        context = "\n".join([f"{m['role'].capitalize()}: {m['content']}" for m in recent_history])

    # System prompt
    system_prompt = (
        "You are BrewGenie, a fun and friendly bartender assistant. "
        "Always suggest drinks, recipes, or nearby liquor stores. "
        "Be conversational, use emojis, and ask follow-up questions like a real bartender. "
        "If the user asks about anything unrelated to alcohol, steer the conversation back to drinks."
    )

    # Combine system prompt, context, and current user query
    full_prompt = f"{system_prompt}\n\n{context}\nUser: {query}\nBrewGenie:"

    # Invoke the RAG chain
    response = qa_chain.invoke({"query": full_prompt})

    # Fallback if response is empty
    result = response.get("result", "").strip()
    if not result:
        return "🤔 Hmm, I'm not sure about that one! Maybe ask for a different type of drink?"

    return result
