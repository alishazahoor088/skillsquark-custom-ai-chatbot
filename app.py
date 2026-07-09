import os
import streamlit as st
from dotenv import load_dotenv

from pdf_loader import load_pdf
from vector_store import create_vector_store
from chatbot import create_chatbot

load_dotenv()

st.set_page_config(page_title="Custom AI Chatbot", page_icon="🤖")

st.title("🤖 Custom AI Chatbot")
st.write("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf"
)

if uploaded_file is not None:

    with open("uploaded.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    documents = load_pdf("uploaded.pdf")
    vector_store = create_vector_store(documents)
    chatbot = create_chatbot(vector_store)

    question = st.text_input("Ask a question about the PDF")

    if question:
        response = chatbot.invoke(question)

        if isinstance(response, dict):
            st.write("### Answer")
            st.write(response.get("result", response))
        else:
            st.write("### Answer")
            st.write(response)
