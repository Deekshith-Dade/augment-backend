from fastapi import APIRouter, Depends, Request
from sqlalchemy.orm import Session
from fastapi.responses import StreamingResponse
from app.database.database import SessionLocal


from langchain_openai import ChatOpenAI

from app.llm_utils.agents.thought_agent.thought_agent import ReactAgent

from app.llm_utils.agents.thought_agent.tools import fetch_relevant_thoughts, get_thought_details


router = APIRouter(prefix="/chat", tags=["chat"])
llm = ChatOpenAI(model="gpt-4o", temperature=0)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

user_id = "83172f77-5d45-4ec2-ac7e-13e3d0f26504"
session_id = "129"

tools = [
    fetch_relevant_thoughts,
    get_thought_details
]
agent = ReactAgent(llm=llm, tools=tools)


@router.post("/")
async def chat(request: Request, db: Session = Depends(get_db)):
    req = await request.json()
    messages = req["messages"]
    question = messages[-1]["content"]
    
    async def stream_response():
        async for chunk in agent.get_stream_response(question, session_id, user_id, db):
            yield chunk
    
    response = StreamingResponse(stream_response(), media_type="text/event-stream")
    response.headers['x-vercel-ai-data-stream'] = 'v1'
    return response