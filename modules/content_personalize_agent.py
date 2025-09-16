import os
import json
from typing import Dict, Any

# LangChain components
from langchain_core.language_models.llms import LLM
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class ContentPersonalizeAgent:
    """
    An agent that inputs a generic syllabus and personalizes it with company-specific details.
    """
    def __init__(self, llm: LLM):
        self.llm = llm
    
        system_prompt = (
                "You are an expert corporate trainer and instructional designer specializing in content personalization. "
                "Your task is to take a generic, pre-written set of language learning modules and rewrite them to be highly specific and relevant to a particular company and employee."
                "\n\n"
                "--- PERSONALIZATION DETAILS ---\n"
                "- Company Name: {company_name}\n"
                "- Products/Services: {products}\n"
                "- Learner's Department: {department}\n"
                "- Typical Interlocutors: {interlocutors}\n"
                "- Common Platforms: {platforms}\n"
                "\n\n"
                "--- INSTRUCTIONS ---\n"
                "1. Read through the entire generic syllabus provided by the user."
                "2. Identify all generic nouns, scenarios, and contexts (e.g., 'a company', 'a product', 'discussing a project with a client', 'writing an email')."
                "3. Systematically replace these generic elements with the specific details provided above. For example, 'a client' might become 'a key partner from the APAC region', and 'a product' could become 'our new 'QuantumLeap' AI platform'."
                "4. **Crucially, do NOT change the structure, learning objectives, or core pedagogical focus of the original units. Only adapt the contextual details to make the materials feel custom-built for an employee at '{company_name}'.**"
                "5. Return the complete, rewritten syllabus in the same Markdown format as the original."
            )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here is the generic syllabus to be personalized: \n\n {module_template}")
        ])

        self.chain = prompt | self.llm | StrOutputParser()

    def run(self, module_template: Dict, personalized_details_path: str) -> Dict[str, Any]:
        pass
        """
        :param
            module_template: generic module generated (output from ModuleGeneratorAgent)
            personalized_details_path: detailed JSON information path, containing tailored information to a target company.
        :return
            a complete, personalized syllabus
        """
        try:
            if not os.path.exists(personalized_details_path):
                raise FileNotFoundError(f"ContentPersonalizeAgent: No personalized JSON information found at {personalized_details_path}")
            with open(personalized_details_path, 'r') as file:
                personalized_detais = json.load(file)

            personalized_module = self.chain.invoke({
                "module_template": module_template,
                **personalized_detais
            })
            return {"personalized_module": personalized_module}
        except Exception as e:
            print(f"ContentPersonalizeAgent: {str(e)}")
            return {"error": str(e)}
