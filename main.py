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

from modules import (
    NeedsAnalysisAgent
)

def run_pipeline():

    # 4. init LLM
    try:
        local_llm = ChatOllama(model="gemma:2b", temperature=0.1, base_url="http://localhost:11434")
        local_embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
        print(f"Loaded language model: {local_llm.model}\n")
        print(f"Loaded embedding model: {local_embedding.model}\n")
    except Exception as e:
        print(f"Failed to load language model. {e}\n")
        return
    
    # 5. init agents
    try:
        need_analysis_agent = NeedsAnalysisAgent(llm=local_llm, embedding_model=local_embedding)
        print("Loaded all agents.\n")
    except Exception as e:
        print(f"Failed to initialize agents. {e}\n")
        return
    
    needs_dir = "/mnt/fa80f336-3342-4d78-8bfd-a43e434a2cda/proj/course-develop/data/jd"
    needs = need_analysis_agent.run(directory_path=needs_dir)

    # 4. Print the result from the dictionary
    print(needs)
    


def main():
    run_pipeline()

if __name__ == "__main__":
    main()