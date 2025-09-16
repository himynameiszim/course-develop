import os

# LangChain components
from langchain_core.language_models.llms import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

class SyllabusDesignerAgent:
    def __init__(self, llm: LLM):
        self.llm = llm

        system_prompt = (
            "You are an expert curriculum designer and ESL/EFL syllabus creator for professional contexts. "
            "Your task is to create a detailed, block-based, skill-focused syllabus of 3 blocks of 10 units, each units of one hour-long session. "
            "based on the provided learner profile. The syllabus should show logical progression, with each block building on the last."
            "\n\n"
            "**Learner Profile:**"
            "- CEFR Level: {cefr_level}"
            "- Needs Analysis: {needs_analysis}"
            "- Context: {context}"
            "\n\n"
            "**Syllabus Requirements:**"
            "1.  **Structure:** THE SYLLABUS MUST BE DIVIDED INTO 3 BLOCKS, EACH BLOCK MUST CONTAIN EXACTLY 10 UNITS."
            "2.  **Unit Detail:** For each Unit, you must specify:"
            "    - **Skill Focus:** The primary skill (e.g., Speaking, Writing, Listening, Reading)."
            "    - **Theme:** A relevant, professional theme for the unit (e.g., 'Leading a Project Kick-off Meeting')."
            "    - **Objectives:** 2-3 clear, measurable 'can-do' objectives for the one-hour session."
            "3.  **Relevance:** All themes and objectives must directly address the learner's profile and needs."
            "4.  **Format:** Use Markdown for clear formatting with main headings for Blocks and sub-headings for Units."
            "\n"
            "NOTE THAT YOU HAVE TO GENERATE THESE FOR EVERY UNIT."
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "Please generate an appropriate 3-blocks x 10 units grammar/vocabulary syllabus now. ")
        ])

        self.chain = prompt | self.llm | StrOutputParser()

    def run(self, cefr_level: str, needs_analysis: str, context: str):
        """
        :param
            cefr_level: the CEFR level range from LevelAnalysisAgent
            needs_analysis: the needs analysis from job description of NeedAnalysisAgent
            context: domain-specific vocabulary, keyword from CorpusAnalysisAgent

        :return
            an appropriate 3 blocks x 10 units grammar/vocabulary syllabus 
        """

        try:
            syllabus = self.chain.invoke({
                "cefr_level": cefr_level,
                "needs_analysis": needs_analysis,
                "context": context
            })
            return str(syllabus)
        except Exception as e:
            return {"error": str(e)}