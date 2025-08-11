import os
import sys
import spacy
import pyinflect
import json
# import warnings

# warnings.filterwarnings("ignore")

from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from keybert import KeyBERT

from modules import (
    NeedsAnalysisAgent,
    CorpusAnalysisAgent
)

def run_pipeline():
    # 1. init LLM
    try:
        local_llm = ChatOllama(model="SOME_MODEL_HERE", temperature=0.1, base_url="http://localhost:11434")
        local_embedding = OllamaEmbeddings(model="SOME_MODEL_HERE")
        local_kw = KeyBERT(model="SOME_MODEL_HERE")
        print(f"Loaded language model: {local_llm.model}\n")
        print(f"Loaded embedding model: {local_embedding.model}\n")
        print(f"Loaded KeyBERT model: {local_kw.model}\n")
    except Exception as e:
        print(f"Failed to load language model. {e}\n")
        return
    
    # 2. init agents
    try:
        need_analysis_agent = NeedsAnalysisAgent(llm=local_llm, embedding_model=local_embedding)
        corpus_analysis_agent = CorpusAnalysisAgent(llm=local_llm, embedding_model=local_embedding, kw_model=local_kw)
        print("Loaded all agents.\n")
    except Exception as e:
        print(f"Failed to initialize agents. {e}\n")
        return
    
    # 3. init directory for processing
    needs_dir = "some directory here"
    corpus_dir = "some directory here"
    needs_result = need_analysis_agent.run(directory_path=needs_dir)
    corpus_result = corpus_analysis_agent.run(directory_path=corpus_dir)

    # 4. print result
    print(needs_result)
    for key, value in corpus_result.items():
        print(f"{key}: {value} \t")
    

def main():
    run_pipeline()

if __name__ == "__main__":
    main()