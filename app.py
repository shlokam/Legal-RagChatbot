from fastapi import FastAPI
from pydantic import BaseModel
from main import decide
app = FastAPI()


# Define request body model
class QueryRequest(BaseModel):
    question: str
    summary: str

@app.post("/decide")
def decide_api(request: QueryRequest):
    return decide(request.question, request.summary)
