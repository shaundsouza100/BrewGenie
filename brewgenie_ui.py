import streamlit as st
from brewgenie_rag_search import get_rag_response


# Page config
st.set_page_config(page_title="🍸 BrewGenie Chatbot", layout="wide")

# Custom CSS
st.markdown("""
    <style>
    body { background-color: #0f0f0f; color: #ffffff; font-family: 'Poppins', sans-serif; }
    .stApp { background-color: #0f0f0f; padding: 2rem; }
    h1 { color: #00AEEF; font-weight: 900; font-size: 3rem; text-align: center; margin-bottom: 2rem; }
    .user-message { background-color: #00598D; color: #fff; padding: 10px 15px; border-radius: 15px; margin: 5px 0; max-width: 80%; align-self: flex-end; }
    .bot-message { background-color: #ffffff; color: #000; padding: 10px 15px; border-radius: 15px; margin: 5px 0; max-width: 80%; align-self: flex-start; box-shadow: 0 0 10px #00598D; }
    .stTextInput > div > div > input { background-color: #1a1a1a; color: #ffffff; border: 2px solid #00598D; border-radius: 20px; padding: 10px; }
    .stTextInput > div > div > input:focus { border-color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1>🍸 BrewGenie Chatbot</h1>", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='user-message'>{msg['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-message'>{msg['content']}</div>", unsafe_allow_html=True)

# Callback function
def handle_submit():
    user_input = st.session_state.input_box
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = get_rag_response(user_input, st.session_state.messages)
    st.session_state.messages.append({"role": "bot", "content": response})
    st.session_state.input_box = ""  # Clear the input inside callback

# Input box with callback
st.text_input("Type your message here:", key="input_box", on_change=handle_submit)
