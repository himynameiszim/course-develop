import os
from typing import Dict, Any
import glob

# LangChain components
from langchain_core.language_models.llms import LLM
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings

# RAG components
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

class ModuleGeneratorAgent:
    """
    An agent to create structured lesson content for each one-hour unit.
    """
    def __init__(self, llm: LLM, embedding_model: Embeddings):
        self.llm = llm
        self.embeddings = embedding_model

    def run(self, cefr_level: str, syllabus_template: str, data_sample_path: str) -> Dict[str, Any]:
        """
        :param
            cefr_level: estimated CEFR level of learner (output from LevelAnalysisAgent)
            syllabus_template: a 3 blocks x 10 units syllabus template (output from SyllabusDesignerAgent)
            data_sample_path: a path containing grammar/vocabulary bank and/or sample corpus extracts for detailed references.
        :return
            a dictionary with keys being block-unit numbers, values are detailed module content.
        """
        try:
            print(f"Loading documents from {data_sample_path}...")
            loader = DirectoryLoader(
                data_sample_path,
                glob="**/*.pdf",
                loader_cls=PyPDFLoader,
                show_progress=True,
                use_multithreading=True
            )
            docs = loader.load()

            if not docs:
                raise FileNotFoundError(f"ModuleGeneratorAgent: No PDF reference data found in '{data_sample_path}'.")

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
            split_docs = text_splitter.split_documents(docs)
            
            vector_store = FAISS.from_documents(split_docs, self.embeddings)
            retriever = vector_store.as_retriever()

            system_prompt = (
                "You are an ESL/EFL instructor. Your task is to generate a detailed, "
                "one-hour lesson plan (module) for a specific unit from a syllabus, using the provided learner profile and given syllabus.\n\n"
                "--- LEARNER PROFILE ---\n"
                "Learner's CEFR level range: {cefr_level}\n"
                "Given syllabus: {syllabus_template}\n\n"
                "--- RETRIEVED MATERIALS ---\n"
                "Use the following materials to help create the 'Input Text/Activity' section:\n{context}\n\n"
                "--- YOUR TASK ---\n"
                "You must now generate the complete lesson module for **Block {block_number}, Unit {unit_number}**.\n\n"
                "--- REQUIRED OUTPUT FORMAT (Strict) ---\n"
                "### Block {block_number}, Unit {unit_number}: [Theme of the unit from the syllabus]\n"
                "**1. Learning Objectives & Outcomes:**\n- [Objective 1 from syllabus]\n- [Objective 2 from syllabus]\n- *Outcome:* By the end of this session, the learner will be able to...\n\n"
                "**2. Input Text/Activity:**\n- [A short reading text, dialogue, grammar explanation, or vocabulary set based on the retrieved materials.]\n\n"
                "**3. Controlled Practice:**\n- [A structured task to practice the target language, e.g., gap-fill, matching, short answer questions.]\n\n"
                "**4. Free Practice:**\n- [A more open-ended, communicative task, e.g., a role-play, a discussion prompt, an email-writing task.]\n\n"
                "**5. Teacher's Notes:**\n- *Instructions:* [Brief, step-by-step instructions for the teacher.]\n- *Model Answers:* [Example answers for the controlled practice task.]"
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "Generate the lesson module for Block {block_number}, Unit {unit_number} now.")
            ])
            rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(self.llm, prompt))

            generated_modules = []
            for block_number in range(1, 4):
                for unit_number in range(1, 11):
                    print(f"Generating Block {block_number}, Unit {unit_number}...")
                    
                    retrieval_query = f"Content for a lesson about Block {block_number}: Unit {unit_number} from the provided syllabus."

                    response = rag_chain.invoke({
                        "input": retrieval_query,
                        "cefr_level": cefr_level,
                        "syllabus_template": syllabus_template,
                        "block_number": block_number,
                        "unit_number": unit_number
                    })
                    generated_modules.append(response["answer"])
                    
            return {"complete_syllabus_modules": "\n\n---\n\n".join(generated_modules)}
        except Exception as e:
            print(f"An error occurred: {e}")
            return {"error": str(e)}