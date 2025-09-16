import os

# LangChain components
from langchain_core.language_models.llms import LLM
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

class LevelAnalysisAgent:
    """
    An agent that determines approximate CEFR level of learners via test data or heuristic estimates via skill descriptions requirements.
    """
    def __init__(self, llm: LLM):
        self.llm = llm

        system_prompt = (
            "You are an expert linguistic analyst and HR language assessment specialist. "
            "Your task is to analyze a job description and determine the required CEFR level range "
            "for a candidate to be successful in the role. "
            "\n\n"
            "Here is a summary of the CEFR levels for your reference:"
            "- C2 (Mastery): Can understand with ease virtually everything heard or read. Can express him/herself spontaneously, very fluently and precisely."
            "- C1 (Advanced): Can understand a wide range of demanding, longer texts. Can express him/herself fluently and spontaneously without much obvious searching for expressions. Can use language flexibly and effectively for social, academic and professional purposes."
            "- B2 (Upper-Intermediate): Can understand the main ideas of complex text on both concrete and abstract topics. Can interact with a degree of fluency and spontaneity that makes regular interaction with native speakers quite possible without strain for either party."
            "- B1 (Intermediate): Can understand the main points of clear standard input on familiar matters. Can deal with most situations likely to arise whilst travelling. Can produce simple connected text on topics which are familiar or of personal interest."
            "- A2 (Elementary): Can understand sentences and frequently used expressions related to areas of most immediate relevance. Can communicate in simple and routine tasks requiring a simple and direct exchange of information."
            "- A1 (Beginner): Can understand and use familiar everyday expressions and very basic phrases."
            "\n\n"
            "Understand all the communicative demands (e.g., negotiating, presenting, technical writing, customer service). "
            "Based on these demands, provide a sufficient CEFR level range."
            "\n\n"
            "CEFR Level: [Your estimated CEFR range, e.g., A2 - B1]"
            "\n\n"
            "ANSWER ONLY WITH A CEFR LEVEL RANGE."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Here is the job description:\n\n{job_description}")
        ])

        # chain: prompt -> llm -> str output
        self.chain = prompt | self.llm | StrOutputParser()

    def run(self, job_description: str) -> str:
        """
        :param
            job_description: string that describes the requirements for a specific job role.
        :return
            a sufficient CEFR level range.
        """
        try:
            range = self.chain.invoke({"job_description": job_description})
            return str(range)
        except Exception as e:
            return f"error: {str(e)}"