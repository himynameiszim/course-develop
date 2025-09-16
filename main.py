# import warnings

# warnings.filterwarnings("ignore")

# LangChain components
from langchain_community.chat_models import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

from keybert import KeyBERT

from modules import (
    NeedsAnalysisAgent,
    CorpusAnalysisAgent,
    LevelAnalysisAgent,
    SyllabusDesignerAgent,
    ModuleGeneratorAgent
)

def run_pipeline():
    # 1. init LLM
    try:
        local_llm = ChatOllama(model="gemma:2b", temperature=0.1, base_url="http://localhost:11434")
        local_embedding = OllamaEmbeddings(model="nomic-embed-text:latest")
        local_kw = KeyBERT(model="all-MiniLM-L6-v2")
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
        level_analysis_agent = LevelAnalysisAgent(llm=local_llm)
        syllabus_designer_agent = SyllabusDesignerAgent(llm=local_llm)
        module_generator_agent = ModuleGeneratorAgent(llm=local_llm, embedding_model=local_embedding)
        print("Loaded all agents.\n")
    except Exception as e:
        print(f"Failed to initialize agents. {e}\n")
        return
    
    # 3. init directory for processing
    needs_dir = "DIRECTORY_PATH"
    corpus_dir = "DIRECTORY_PATH"
    level_dir = "DIRECTORY_PATH"
    data_bank = "DIRECTORY_PATH"

    # 4. main pipeline
    try:
        print("-----Running Need Analysis Agent-----\n")
        needs_result = need_analysis_agent.run(directory_path=needs_dir)
        print(needs_result + "\n")
    except Exception as e:
        print("Error during Need Analysis Agent\n")
        return str(e)

    try:
        print("-----Running Corpus Analysis Agent-----\n")
        corpus_result = corpus_analysis_agent.run(directory_path=corpus_dir)
        for key, value in corpus_result.items():
            print(f"{key}: {value}\t\n")
    except Exception as e:
        print("Error during Corpus Analysis Agent\n")
        return str(e)

    try:
        print("-----Running CEFR Level Analysis Agent-----\n")
        level_result = level_analysis_agent.run(job_description=needs_result)
        print(level_result + "\n")
    except Exception as e:
        print("Error during Level Analysis Agent\n")
        return str(e)
    
    try:
        print("-----Running Syllabus Designer Agent-----\n")
        syllabus_result = syllabus_designer_agent.run(cefr_level=level_result, needs_analysis=needs_result, context=corpus_result)
        print(syllabus_result + "\n") 
    except Exception as e:
        print("Error during Syllabus Designer Agent\n")
        return str(e)
    
    try:
        print("-----Running Module Generator Agent-----\n")
        module_result = module_generator_agent.run(cefr_level=level_result, syllabus_template=syllabus_result, data_sample_path=data_bank)
        print(module_result)
    except Exception as e:
        print(f"Error during Module Generator Agent\n{e}")
        return str(e)

    
def main():
    run_pipeline()

if __name__ == "__main__":
    main()