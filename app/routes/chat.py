import json
from fastapi import APIRouter, Depends, Request, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_async_db
import uuid
from datetime import datetime, timezone

from langchain_openai import ChatOpenAI

from app.llm_utils.tags import generate_title
from app.llm_utils.agents.thought_agent.thought_agent import ReactAgent
from app.llm_utils.agents.thought_agent.tools import fetch_relevant_thoughts, get_thought_details, web_search_tool
from app.schemas.schemas import ChatSessionResponse
from app.models.models import ChatSession, User
from app.routes.utils import get_current_user

router = APIRouter(prefix="/chat", tags=["chat"])
llm = ChatOpenAI(model="gpt-4o", temperature=0)


tools = [
    fetch_relevant_thoughts,
    get_thought_details,
    web_search_tool
]
agent = ReactAgent(llm=llm, tools=tools)


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def get_chat_sessions(db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    try:
        stmt = select(ChatSession).where(ChatSession.user_id == user.id)
        sessions = await db.execute(stmt)
        sessions = sessions.scalars().all()
        sessions_response = []
        for session in sessions:
            sessions_response.append(ChatSessionResponse(id=str(session.id), title=session.title, created_at=session.created_at.isoformat(), updated_at=session.updated_at.isoformat()))
        return sessions_response
        
    except Exception as e:
        print(e)
        raise e

@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    try:
        stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
        session = await db.execute(stmt)
        session = session.scalar_one_or_none()
        await db.delete(session)
        await db.commit()
        return {"message": "Session deleted successfully"}
    except Exception as e:
        print(e)
        raise e

@router.get("/history/{session_id}")
async def get_session_history(session_id: str, db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    try:
        stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
        session = await db.execute(stmt)
        session = session.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = await agent.get_session_history(session_id, user.id)
        return messages
    except Exception as e:
        print(e)
        return HTTPException(status_code=500, detail="Internal server error")


@router.post("/")
async def chat(request: Request, db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    req = await request.json()
    try:
        messages = req["messages"]
        question = messages["content"]
        session_id = req["session_id"]
        
        if not session_id:
            session_id = str(uuid.uuid4())
            title = generate_title(question)
            current_time = datetime.now(timezone.utc)
            session = ChatSession(
                id=session_id,
                user_id=user.id,
                title=title,
                created_at=current_time,
                updated_at=current_time
            )
            db.add(session)
            await db.commit()
            await db.refresh(session)
        else:
            stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
            session = await db.execute(stmt)
            session = session.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            
        async def stream_response():
            try:
                data = [{"session": {"id": str(session.id), "title": session.title, "created_at": session.created_at.isoformat(), "updated_at": session.updated_at.isoformat()}}]
                yield f"8:{json.dumps(data)}\n"
                async for chunk in agent.get_stream_response(question, session.id, user.id, db):
                    yield chunk
                    
            except Exception as e:
                print(e)
                yield f"3:{json.dumps('error')}\n"
        response = StreamingResponse(stream_response(), media_type="text/event-stream")
        response.headers['x-vercel-ai-data-stream'] = 'v1'
        return response
    except Exception as e:
        print(e)
        raise e 