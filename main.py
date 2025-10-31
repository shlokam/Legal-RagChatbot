from ragPipeline import llm, rag_retriever, AdvancedRAGPipeline
from langchain.prompts import PromptTemplate
import re

def llmPrompt(question: str, summary : str):
  print("prompting first lllm")
  prompt = f""" You are an expert in Indian Income Tax Law. You need to decide what tool to use to respond.

conversation_summary: {summary}

- Use the tool 'direct answer' if the question is general, conceptual, explanatory, or refers to a section/provision that you already know about (even if the answer is brief or high-level).
- Use the tool 'RAG' only if the question requires specific case law, tribunal decisions, latest amendments, or detailed judicial precedents that you cannot reliably answer from your own knowledge.

Question: {question}

What tool do you want to use? Only reply with given the below format. Include nothing else.
Answer format:

### Tool Choice:
Either 'direct answer' or 'RAG'

### Optimized Query:
If you chose 'RAG', transform the user's question into an optimized vector search query using these principles:

OBJECTIVE:
Your task is to transform raw user questions into dense, context-rich, and statute-specific legal search queries optimi

Process You Must Follow
Step 1 — Deep Legal Understanding
Analyze the user question to extract:
The core statute / section / act / rule / case type (e.g., Section 10 of the Income Tax Act, 1961).
The specific aspect being asked (definition, exemption, condition, procedure, penalty, interpretation, etc.).
The intent behind the question — whether the user seeks judicial interpretation, statutory meaning, applicability, scope, or historical evolution.
Distinguish this section from similar ones (e.g., Section 10 vs Section 14 or 17).

Step 2 — Context Enrichment
Add statute-relevant details that make the search semantically strong:
What this section governs (e.g., “income not included in total income”).
Who it applies to (e.g., salaried individuals, agricultural earners).
Common judicial contexts where this section is debated (e.g., “eligibility for HRA exemption,” “scope of agricultural income”).
Include common case law triggers (e.g., “misuse of exemption,” “interpretation conflict,” “scope limitation”).

Step 3 — Query Construction
Construct a single paragraph optimized for dense semantic search, following these rules:
Use exact statutory names (e.g., Income Tax Act, 1961) and keywords appearing in judgments (e.g., “interpreted,” “held,” “ruled,” “scope,” “applicability”).
Avoid conversational phrases (“show me,” “I want”).
Include jurisdictional hints (“Indian High Court,” “Supreme Court,” “Income Tax Appellate Tribunal”).
Capture intent clarity — e.g., whether the user wants interpretations, applicability, comparisons, or historical references.
Avoid metadata-like filters (no “date,” “court,” “judge,” unless explicitly asked).

Output Format
Output only the final optimized query in one paragraph, not the reasoning steps.
Ensure the query is:
Statute-specific
Semantically dense
Contextually rich
Optimized for retrieval accuracy (not keyword search)

                      """
 
  response = llm.invoke([prompt.format(conversation_summary=summary, question=question)])
  answer = response.content

  # print("llm first response " + answer)
  return answer 

def decide(question: str, summary : str):
  llm_answer = llmPrompt(question, summary) 
  print(llm_answer)
  
  tool_pattern = r'### Tool Choice:\s*\n\s*([^\n]+)'
  tool_match = re.search(tool_pattern, llm_answer)
  tool = tool_match.group(1).strip() if tool_match else ""

  question_pattern = r'### Optimized Query:\s*\n\s*"?([^"]+)"?'
  question_match = re.search(question_pattern, llm_answer)
  question_tool = question_match.group(1).strip() if question_match else ""

  print("Tool:", tool)
  print("Question:", question_tool)
  
  if "rag" in tool.strip().lower():
    print('inside llm anseer Rag and calling rag retreive')
    adv_rag = AdvancedRAGPipeline(rag_retriever, llm)
    result = adv_rag.query(question_tool, summary, top_k=10, min_score=0.3, stream=True, summarize=True)
    print("\nFinal Answer:", result['answer'])
    return result
    return []
  else:
    prompt = f""" You are an expert in Indian Income Tax Law. Also, attaching the previous conversation summary.

                      Question:
                      {question} 
                      
                      conversation_summary : {summary}

                      In the end include the below section.

                      ### Summary of the Answer:
                      Provide a brief summary of the question asked and answer you have given above in maximum 5 sentences .
                      """
 
    response = llm.invoke([prompt.format(question=question, conversation_summary=summary)])
    answer = response.content

    summary_pattern = r'### Summary of the Answer:\s*\n(.*?)\s*(?:### List of Referred PDFs:|$)'
    summary_match = re.search(summary_pattern, answer, re.DOTALL)
    summary_answer = summary_match.group(1).strip() if summary_match else ""

    return {
            'question': question,
            'answer': answer,
            'sources': [],
            'summary': summary_answer
        }