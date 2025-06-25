from fastapi import APIRouter, HTTPException
from fastapi import Request, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import logger
from app.routes.utils import get_current_user
from app.models.models import User
from app.database.database import get_async_db


from langchain_openai import ChatOpenAI
from app.llm_utils.agents.flow_agent.flow_agent import FlowAgent


router = APIRouter(prefix="/flow", tags=["flow"])

llm = ChatOpenAI(model="gpt-4o", temperature=0)

flow_agent = FlowAgent(llm=llm, max_themes=4, max_emotions=4, max_goals=4)


@router.post("/")
async def flow(request: Request, db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    logger.info("flow_request", user_id=user.id)
    req = await request.json()
    try:
        message = req["message"]
        response = await flow_agent.get_response(message, user.id, db)
        return response
    except Exception as e:
        logger.error("flow_request_error", user_id=user.id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))