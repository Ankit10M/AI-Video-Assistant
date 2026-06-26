from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_core.documents import Document

CHROMA_DB = "vector_db"
COLLECTION_NAME = "meeting_transcripts"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={"device": "cpu"})

def build_vector_store(transcript:str)-> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(transcript)
    docs = [
        Document(page_content=chunk, metadata={'chunk_index': i})
        for i, chunk in enumerate(chunks)
    ]
    embeddings = get_embeddings()
    vector_store = Chroma.from_documents(
        documents = docs,
        embedding = embeddings,
        collection_name = COLLECTION_NAME,
        persist_directory = CHROMA_DB
    )
    return vector_store

def load_vector_store() -> Chroma:
    embeddings = get_embeddings()
    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=CHROMA_DB,
        embedding_function=embeddings
    )
    return vector_store

def get_retriever(vectore_store: Chroma, k:int=4):
    return vectore_store.as_retriever(search_type = 'similarity', search_kwargs={"k": k})
