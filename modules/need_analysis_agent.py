import os
import glob
from typing import Dict

from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class NeedsAnalysisAgent:
    """
    An agent that analyzes PDFs to extract communicative tasks.
    Initialized with a language model and an embedding model.
    """
    def __init__(self, llm: LLM, embedding_model: Embeddings):
        self.llm = llm
        self.embeddings = embedding_model

    def run(self, directory_path: str) -> str:
        """
        Executes the analysis on a directory of PDFs and returns a structured dictionary.

        This method builds and executes a temporary RAG pipeline for the given data.
        
        Args:
            directory_path: The path to the directory containing PDF files.

        Returns:
            A dictionary containing the analysis report.
        """
        # Step 1: Load and process documents
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{directory_path}'.")

        docs = [PyPDFLoader(pdf_path).load() for pdf_path in pdf_files]
        flat_docs = [doc for sublist in docs for doc in sublist]
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
        split_docs = text_splitter.split_documents(flat_docs)
        
        # Step 2: Create a data-specific vector store and retriever
        vector_store = FAISS.from_documents(split_docs, self.embeddings)
        retriever = vector_store.as_retriever()

        # Step 3: Define the prompt and build the RAG chain for this specific run
        system_prompt = (
            "You are an expert linguistic and occupational analyst. Your task is to analyze the "
            "provided context from job descriptions, questionnaires, and other documents to identify "
            "all tasks that require significant communication skills."
            "\n\n"
            "Use the following retrieved context to answer the query."
            "\n\n"
            "For each distinct communicative task you identify, structure your output exactly as follows:"
            "\n\n"
            "### Communicative Task:"
            "**- Discourse Functions:**"
            "**- Register/Domain:**"
            "\n---"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages(
            [("system", system_prompt), ("human", "{input}")]
        )
        
        # Using LangChain's helper function is the modern way to build the chain
        Youtube_chain = create_stuff_documents_chain(self.llm, prompt)
        rag_chain = create_retrieval_chain(retriever, Youtube_chain)

        # Step 4: Invoke the chain and get the result
        response = rag_chain.invoke(
            {"input": "Based on the provided documents, produce a structured list of all communicative tasks."}
        )
        return response["answer"]