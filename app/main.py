from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from app.routes import thoughts, chat, articles, webhooks, flow
from fastapi.middleware.cors import CORSMiddleware
from app.core.logging import logger
from app.core.limiter import limiter, rate_limits
from app.utils.utils import parse_list_from_env


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("application_startup", project_name="augment")
    yield
    logger.info("application_shutdown", project_name="augment")

app = FastAPI(lifespan=lifespan)
app.include_router(thoughts.router)
app.include_router(chat.router)
app.include_router(articles.router)
app.include_router(webhooks.router)
app.include_router(flow.router)




app.add_middleware(
    CORSMiddleware,
    allow_origins=parse_list_from_env("ALLOWED_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
@limiter.limit(rate_limits["RATE_LIMIT_DEFAULT"][0])
def hello_word(request: Request):
    logger.info("hello_world_request")
    return {"text": "hello world"}