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
from app.core.logging import logger
from app.core.limiter import limiter, rate_limits

router = APIRouter(prefix="/chat", tags=["chat"])
llm = ChatOpenAI(model="gpt-4o", temperature=0)


tools = [
    fetch_relevant_thoughts,
    get_thought_details,
    web_search_tool
]
agent = ReactAgent(llm=llm, tools=tools)


@router.get("/sessions", response_model=list[ChatSessionResponse])
@limiter.limit(rate_limits["RATE_LIMIT_SESSIONS_READ"][0])
async def get_chat_sessions(request: Request, db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    logger.info("get_chat_sessions_request", user_id=user.id)
    try:
        stmt = select(ChatSession).where(ChatSession.user_id == user.id)
        sessions = await db.execute(stmt)
        sessions = sessions.scalars().all()
        sessions_response = []
        for session in sessions:
            sessions_response.append(ChatSessionResponse(id=str(session.id), title=session.title, created_at=session.created_at.isoformat(), updated_at=session.updated_at.isoformat()))
        logger.info("get_chat_sessions_request_return", user_id=user.id, len_sessions=len(sessions_response))
        return sessions_response
        
    except Exception as e:
        logger.error("get_chat_sessions_request_error", user_id=user.id, error=str(e))
        raise e

@router.delete("/sessions/{session_id}")
@limiter.limit(rate_limits["RATE_LIMIT_SESSION_DELETE"][0])
async def delete_session(request: Request, session_id: str, db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    logger.info("delete_session_request", user_id=user.id, session_id=session_id)
    try:
        stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
        session = await db.execute(stmt)
        session = session.scalar_one_or_none()
        await db.delete(session)
        await db.commit()
        logger.info("delete_session_request_return", user_id=user.id, session_id=session_id)
        return {"message": "Session deleted successfully"}
    except Exception as e:
        logger.error("delete_session_request_error", user_id=user.id, error=str(e))
        raise e

@router.get("/history/{session_id}")
@limiter.limit(rate_limits["RATE_LIMIT_SESSION_HISTORY"][0])
async def get_session_history(request: Request, session_id: str, db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    logger.info("get_session_history_request", user_id=user.id, session_id=session_id)
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required")
    try:
        stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
        session = await db.execute(stmt)
        session = session.scalar_one_or_none()
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        messages = await agent.get_session_history(session_id, user.id)
        logger.info("get_session_history_request_return", user_id=user.id, session_id=session_id)
        return messages
    except Exception as e:
        logger.error("get_session_history_request_error", user_id=user.id, session_id=session_id, error=str(e))
        return HTTPException(status_code=500, detail="Internal server error")


@router.post("/")
@limiter.limit(rate_limits["RATE_LIMIT_CHAT"][0])
async def chat(request: Request, db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    logger.info("chat_request", user_id=user.id)
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
            logger.info("chat_request_new_session_commit", user_id=user.id, session_id=session_id)
        else:
            stmt = select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user.id)
            session = await db.execute(stmt)
            session = session.scalar_one_or_none()
            if not session:
                raise HTTPException(status_code=404, detail="Session not found")
            logger.info("chat_request_existing_session", user_id=user.id, session_id=session_id)
        async def stream_response():
            try:
                data = [{"session": {"id": str(session.id), "title": session.title, "created_at": session.created_at.isoformat(), "updated_at": session.updated_at.isoformat()}}]
                yield f"8:{json.dumps(data)}\n"
                async for chunk in agent.get_stream_response(question, session.id, user.id, db):
                    yield chunk
                    
            except Exception as e:
                logger.error("chat_request_error", user_id=user.id, error=str(e))
                yield f"3:{json.dumps('error')}\n"
        response = StreamingResponse(stream_response(), media_type="text/event-stream")
        response.headers['x-vercel-ai-data-stream'] = 'v1'
        logger.info("chat_request_return", user_id=user.id, session_id=session_id)
        return response
    except Exception as e:
        logger.error("chat_request_error", user_id=user.id, error=str(e))
        raise e 