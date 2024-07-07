from langchain_community.vectorstores.chroma import Chroma
from langchain_openai import OpenAIEmbeddings
import os

class MyVectorStore:
    def __init__(self):
        pass

    def embed_text_and_return_vectorstore(self, text_chunks):
        openapi_key = os.getenv("OPENAI_API_KEY")
        embeddings = OpenAIEmbeddings()
        vectorstore = Chroma.from_texts(text_chunks, embeddings)
        return vectorstore

    def get_retriever(self, vectorstore):
        retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        return retriever

