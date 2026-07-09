from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS


def create_vector_store(documents):
    """
    Create a FAISS vector store from PDF documents.
    """
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )

    vector_store = FAISS.from_documents(
        documents,
        embeddings
    )

    return vector_store
