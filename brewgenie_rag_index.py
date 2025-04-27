from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load FAISS index
vectorstore = FAISS.load_local("brewgenie_faiss_index", OpenAIEmbeddings(model="text-embedding-ada-002"), allow_dangerous_deserialization=True)

# Initialize OpenAI GPT model
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

# Create RetrievalQA chain
retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Define the function to use in Streamlit
def get_rag_response(query):
    response = qa_chain.invoke(query)
    return response['result']  # Adjust based on actual output structure
