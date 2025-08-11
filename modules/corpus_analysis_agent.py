import os
import glob
from typing import Dict, List, Any

# Core LangChain components
from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# RAG components for the LLM part
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# Keyword extraction component
from keybert import KeyBERT

class CorpusAnalysisAgent:
    """
    An agent that analyzes documents to extract domain-specific vocabulary
    (using KeyBERT) and identify structural templates (using an LLM with RAG).
    """
    def __init__(self, llm: LLM, embedding_model: Embeddings, kw_model: KeyBERT):
        """
        Initializes the agent with its LLM and embedding dependencies.

        Args:
            llm: An initialized LangChain language model.
            embedding_model: An initialized LangChain embedding model.
        """
        self.llm = llm
        self.embeddings = embedding_model
        self.kw_model = kw_model

    def _get_text_from_directory(self, directory_path: str) -> str:
        """Loads and concatenates all text from PDFs in a directory."""
        pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
        if not pdf_files:
            raise FileNotFoundError(f"No PDF files found in '{directory_path}'.")
        
        full_text = []
        for pdf_path in pdf_files:
            loader = PyPDFLoader(pdf_path)
            pages = loader.load_and_split()
            for page in pages:
                full_text.append(page.page_content)
        
        return "\n".join(full_text)
    
    def _extract_vocabulary(self, full_text: str) -> List[Dict[str, Any]]:
        """Extracts keywords using KeyBERT."""
        keywords = self.kw_model.extract_keywords(
            full_text,
            keyphrase_ngram_range=(1, 3), # Consider phrases of up to 3 words
            stop_words='english',
            use_mmr=True, # Max sum similarity for diversity
            diversity=0.7,
            top_n=30 # Top-k
        )
        # Formatting
        return [{"keyword": kw, "relevance": round(score, 4)} for kw, score in keywords]

    def _analyze_structure(self, directory_path: str) -> str:
        """Analyzes document structure using the LLM with a RAG pipeline."""
        # RAG pipeline
        docs = [PyPDFLoader(pdf_path).load() for pdf_path in glob.glob(os.path.join(directory_path, "*.pdf"))]
        flat_docs = [doc for sublist in docs for doc in sublist]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(flat_docs)
        
        vector_store = FAISS.from_documents(split_docs, self.embeddings)
        retriever = vector_store.as_retriever()

        system_prompt = (
            "You are a professional document analyst and tech expert. "
            "Your task is to analyze the provided excerpts from contracts or job documents. "
            "Identify domain-specific vocabulary (frequency-based or keyword-based) and also structural templates of the given documents. "
            "Synthesize this into a general template or a list of common sections."
            "\n\n"
            "Focus on the high-level structure, such as 'Preamble', 'Definitions Section', 'Scope of Work', etc... "
            "DO NOT just repeat the text; describe the purpose of each section."
            "\n\n"
            "CONTEXT:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        
        rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(self.llm, prompt))
        
        response = rag_chain.invoke({"input": "Analyze the structure of these documents and provide a template."})
        return response["answer"]

    def run(self, directory_path: str) -> Dict[str, Any]:
        """
        Executes the full analysis pipeline.

        Args:
            directory_path: Path to the directory with documents.

        Returns:
            A dictionary with domain vocabulary and structural analysis.
        """
        full_text = self._get_text_from_directory(directory_path)
        vocabulary = self._extract_vocabulary(full_text)
        structure = self._analyze_structure(directory_path)
        
        return {
            "domain_vocabulary": vocabulary,
            "structural_analysis": structure
        }