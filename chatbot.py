from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA


def create_chatbot(vector_store):
    """
    Create a RetrievalQA chatbot using Gemini.
    """
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.3
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

    return qa_chain
