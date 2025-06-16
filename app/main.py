from fastapi import FastAPI
from app.routes import thoughts, chat, articles, webhooks
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.include_router(thoughts.router)
app.include_router(chat.router)
app.include_router(articles.router)
app.include_router(webhooks.router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def hello_word():
    return {"text": "hello world"}