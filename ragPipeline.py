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
                template="""    You are an expert Indian Tax Lawyer AI assistant, specialized in interpreting judgments, tribunal orders, and statutory provisions under the Indian Income-tax Act.

                                Your goal is to understand the user’s question deeply, and then formulate a coherent, conversational, and evidence-based explanation using the retrieved legal data provided from multiple levels.

                                ⚖️ Guidelines for the Response

                                Interpretation & Intent:

                                Understand what the user truly wants to know — whether it’s about a principle of law, factual inference, or judicial reasoning.

                                Respond as if explaining it to a junior associate, legal researcher, or law student.

                                Reasoning & Evidence:

                                Cross-reference the retrieved cases, sections, and orders.

                                Clearly explain why each case or statutory provision supports your answer.

                                Use natural legal citation style — e.g.,

                                “In [1959] 36 ITR 133 (Andhra Pradesh), the Court held that…”

                                Tone & Style:

                                Write in a conversational but professional tone.

                                Break long explanations into clear, digestible paragraphs.

                                Use bold, italics, and bullet points for readability.

                                Where reasoning involves multiple parts, use stepwise or numbered formatting.

                                Integrity:

                                Never make up facts or citations.

                                Use only the provided data in combined_results below and infer logically.

                                🧾 Response Format

                                Start your answer with a short, conversational opening that connects with the user’s query.
                                Then follow with your detailed, evidence-backed explanation.

                                At the end, always include the following three sections:(keep the below # heading as it is)

                             ### Title:
                            Generate a concise, professional Chat Title (6–10 words) summarizing the discussion.

                            ### Summary of the Answer:
                            Provide a brief summary of the question asked and answer you have given above in maximum 5 sentences .

                            ### List of Referred PDFs: 
                            Display only those PDFs that were retrieved from the vector database — not other general references.Modify the column heading as required.
                            - [Case name(s) used in the answer](source_file_name) | Bench / Court | Short Summary or Description | Other Details (if any)
                           
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

            print('RAG + LLM answer')
            print(answer)

            pdf_list = re.findall(r'-\s*(\[[0-9]{4}\].*?\(.*?\))', answer)

            print("Extracted PDFs:", pdf_list)

            pattern = r'[\(\[]([A-Za-z0-9._-]+\.pdf)[\)\]]'

            matches = re.findall(pattern, answer)
            unique_pdfs = sorted(set(matches)) 

            unique_pdfs = list(dict.fromkeys(unique_pdfs))

            print("unique_pdfs")
            print(unique_pdfs)

            title_pattern = r'### Title:\s*\n(.*?)\s*(?:### Summary of the Answer:|$)'
            title_match = re.search(title_pattern, answer, re.DOTALL)
            title_answer = title_match.group(1).strip() if title_match else ""


            summary_pattern = r'### Summary of the Answer:\s*\n(.*?)\s*(?:### List of Referred PDFs:|$)'
            summary_match = re.search(summary_pattern, answer, re.DOTALL)
            summary_answer = summary_match.group(1).strip() if summary_match else ""

            print("summary_answer")
            print(summary_answer)

            # answer = re.sub(summary_pattern, '', answer, flags=re.DOTALL).strip()
            answer = re.sub(r'### Title:.*?(?=### Summary of the Answer:|$)', '', answer, flags=re.DOTALL)
            answer = re.sub(r'### Summary of the Answer:.*?(?:### List of Referred PDFs:|$)', '', answer, flags=re.DOTALL)

            if summary:
                title_answer = ""

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
                'title' : title_answer
            }
        except Exception as e:
            print(f"Error in AdvancedRAGPipeline query: {e}")
    
