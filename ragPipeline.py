# --- Advanced RAG Pipeline: Streaming, Citations, History, Summarization ---
from typing import  Dict, Any
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from ragRetreiver import RAGRetriever
from embedding import EmbeddingManager
from vectorStore_AWS import VectorStore
from langchain.schema import HumanMessage
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import re
from bs4 import BeautifulSoup
from markdown import markdown
import datetime
from configuration import load_env, get_api_key

load_env()
groq_api_key = get_api_key()

llm=ChatGroq(groq_api_key=groq_api_key,model_name="meta-llama/llama-4-maverick-17b-128e-instruct",temperature=0.1,max_tokens=1024)
embedding_manager = EmbeddingManager()
vectorstore = VectorStore()
rag_retriever=RAGRetriever(vectorstore,embedding_manager)


class AdvancedRAGPipeline:
    def __init__(self, retriever, llm):
        self.retriever = rag_retriever
        self.llm = llm
        self.history = []
        

    def query(self, question: str, summary:str, top_k: int = 5, min_score: float = 0.5, stream: bool = False, summarize: bool = False) -> Dict[str, Any]:
        print("Inside AdvancedRAGPipeline query method")
        try:
            results = self.retriever.retrieve(question, score_threshold=min_score, top_k=top_k)
            print("conversation_summary")
            print(summary)

            prompt_template = PromptTemplate(
                input_variables=["combined_results", "content"],
                template="""  You are an expert Indian Tax Lawyer AI assistant specialized in interpreting judgments, tribunal orders, and statutory provisions under the Indian Income-tax Act.

                            Your goal is to understand the user’s question deeply, then formulate a coherent, conversational, and evidence-based response using the retrieved data provided from multiple levels.

                            {conversation_summary}


                            Your Objectives:
                            - Interpret the user’s query like a human legal expert — infer what the user is truly asking (intent).
                            - Cross-reference the provided data — reason about which parts best answer the query.
                            - Respond conversationally, as if you’re explaining it to a legal researcher or law student.
                            - Correlate the evidence explicitly — explain why each cited case or section supports the answer.
                            - **When citing cases, refer to them naturally (e.g., “In [1959] 36 ITR 133 (Andhra Pradesh)…”), not using technical labels like ‘key’ or ‘source_file’.**
                            - Never make up or assume data — rely strictly on the retrieved information.
                            - End with a concise conclusion that logically answers the query.

                            In the end include the below 2 sections.

                            ### Summary of the Answer:
                            Provide a brief summary of the question asked and answer you have given above in maximum 5 sentences .

                            ### List of Referred PDFs:
                            - [Case name(s) used in the answer](source_file_name)
                            User Query:
                            {content}

                            Retrived Results from Vector DB:
                            {combined_results}

                            """        )
            print("Formatting prompt and Caling llm")

            formatted_prompt = prompt_template.format(conversation_summary = summary, combined_results=results, content=question)
            messages = [HumanMessage(content=formatted_prompt)]

            response = llm.invoke(messages)
            answer = response.content

            pdf_list = re.findall(r'-\s*(\[[0-9]{4}\].*?\(.*?\))', answer)

            print("Extracted PDFs:", pdf_list)

            pattern = r'[\(\[]([A-Za-z0-9._-]+\.pdf)[\)\]]'

            matches = re.findall(pattern, answer)
            unique_pdfs = sorted(set(matches)) 

            unique_pdfs = list(dict.fromkeys(unique_pdfs))

            print("unique_pdfs")
            print(unique_pdfs)

            summary_pattern = r'### Summary of the Answer:\s*\n(.*?)\s*(?:### List of Referred PDFs:|$)'
            summary_match = re.search(summary_pattern, answer, re.DOTALL)
            summary_answer = summary_match.group(1).strip() if summary_match else ""

            print("summary_answer")
            print(summary_answer)

            source_file_list = []

            for vec in results:
                meta = vec.get("metadata", {})
                case_meta = meta.get("case_metadata", [])

                if len(case_meta) >= 4:
                
                    court_location = case_meta[1]
                    date_str = case_meta[2]
                    source_file = case_meta[3]

                    if source_file in unique_pdfs:
                        try:
                            date_obj = datetime.datetime.strptime(date_str, "%Y-%m-%d")
                            year = date_obj.strftime("%Y")
                            month = date_obj.strftime("%m")
                            path = f"{court_location}/{year}/{month}/{source_file}"
                            source_file_list.append(path)
                        except ValueError:
                            pass

            print("source_file_list" )
            print(source_file_list)

            return {
                'question': question,
                'answer': answer,
                'sources': source_file_list,
                'summary': summary_answer,
            }
        except Exception as e:
            print(f"Error in AdvancedRAGPipeline query: {e}")
    



