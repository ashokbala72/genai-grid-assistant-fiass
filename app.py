import streamlit as st
from retriever_chain import build_chain

st.title("GenAI Grid Assistant")

qa = build_chain()

query = st.text_input("Enter your question:")
if query:
    response = qa.run(query)
    st.write("🔍 Answer:")
    st.write(response)