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
    # print("\nFinal Answer:", result['answer'])
    return result
  else:
    prompt = f""" You are an expert in Indian Income Tax Law. Also, attaching the previous conversation summary. If no summary is given, assume this is the first chat with the user.
                  Don't add this is first chat or no summary provided before in the response.

                  Interpretation & Intent:
                    - Understand what the user truly wants to know — whether it’s about a principle of law, factual inference, or judicial reasoning.
                    - Respond as if explaining it to a junior associate, legal researcher, or law student.
                  Tone & Style:
                      -Write in a conversational but professional tone.
                      -Break long explanations into clear, digestible paragraphs.
                      -Use bold, italics, and bullet points for readability.
                      -Where reasoning involves multiple parts, use stepwise or numbered formatting.

                    💬 Response Requirements
                          -Provide a coherent, evidence-based legal explanation for the given question.
                          -End the main answer with a natural follow-up question, without explicitly labeling it as “Follow up question”.
                          -Example: “Would you like me to retrieve relevant case laws that support this interpretation?”
                          -In the summary section, include that same follow-up question prefixed with “Follow up question:”.

                      Question:
                      {question} 
                      
                      conversation_summary : {summary}

                      Add a follow-up question based on user query and if you think cases can be found for the query then add the follow up question whether user wants us to retreive cases for it. Like a coversational chatbot.
                      In the end include the below sections title and summary. Also, add the follow up question in summary.  Only explicitly add Follow up question : '' in summary part.

                      ### Title:
                        Generate a concise, professional Chat Title (6–10 words) summarizing the discussion.

                      ### Summary of the Answer:
                      Provide a brief summary (maximum 5 sentences) covering:
                        - The essence of the question asked,
                        - The reasoning and conclusion you’ve given,
                        - And finally, the Follow up question: "<your follow-up>"
                      """
 
    response = llm.invoke([prompt.format(question=question, conversation_summary=summary)])
    answer = response.content

    print(answer)

    title_pattern = r'### Title:\s*\n(.*?)\s*(?:### Summary of the Answer:|$)'
    title_match = re.search(title_pattern, answer, re.DOTALL)
    title_answer = title_match.group(1).strip() if title_match else ""

    # answer = re.sub(title_answer, '', answer, flags=re.DOTALL).strip()

    summary_pattern = r'### Summary of the Answer:\s*\n(.*?)\s*(?:### List of Referred PDFs:|$)'
    summary_match = re.search(summary_pattern, answer, re.DOTALL)
    summary_answer = summary_match.group(1).strip() if summary_match else ""
    
    # answer = re.sub(summary_answer, '', answer, flags=re.DOTALL).strip()

    answer = re.sub(r'### Title:.*?(?=### Summary of the Answer:|$)', '', answer, flags=re.DOTALL)
    answer = re.sub(r'### Summary of the Answer:.*?(?:### List of Referred PDFs:|$)', '', answer, flags=re.DOTALL)

    if summary:
      title_answer = ""


    return {
            'question': question,
            'answer': answer,
            'sources': [],
            'summary': summary_answer,
            'title' : title_answer
        }
