import sys
sys.path.append("..")

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from sqlalchemy import text
from app.embeddings.embeddings import embed_text_openai
from app.models.models import Thought

from dotenv import load_dotenv
import os
load_dotenv()

def pretty_thoughts(results):
    texts = []
    for result in results:
        id, title, full_content, distance = result
        texts.append(f"### ID:{id} \n\n ### Title:{title}({-distance:.2f})\n\nContent:\n{full_content[:100]}")
    final_text = "\n\n".join(texts)
    return final_text


def get_similar_thoughts(db, query_embedding, user_id, top_k=5):
    sql = text("""
        SELECT id, title, full_content, embedding <#> (:embedding)::vector AS distance
        FROM thoughts
        WHERE user_id = :user_id
        ORDER BY embedding <#> (:embedding)::vector
        LIMIT :top_k
    """)
    result = db.execute(
        sql,
        {
            "embedding": query_embedding,
            "user_id": user_id,
            "top_k": top_k
        }
    ).fetchall()
    return result


def to_pgvector(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"

@tool
async def fetch_relevant_thoughts(query: str, config: RunnableConfig) -> str:
    """A Rag Tool that based on a "descriptive" query provided fetches relevant thoughts that the user has posted"""
    try:
        user_id = config.get("configurable").get("user_id")
        session = config.get("configurable").get("session")
    except Exception as e:
        raise("User ID not provided")
    
    embedding = embed_text_openai(query)
    pg_vector_str = to_pgvector(embedding)
    response = get_similar_thoughts(session, pg_vector_str, user_id, top_k=5)
    result = pretty_thoughts(response)
    return result


@tool
async def get_thought_details(thought_id: str, config: RunnableConfig) -> str:
    """A tool that fetches the information about particular thought requested"""
    try:
        user_id = config.get("configurable").get("user_id")
        session = config.get("configurable").get("session")
    except Exception as e:
        raise("User ID not provided")
    
    thought = session.query(Thought).filter_by(id=thought_id, user_id=user_id).first()
    if not thought:
        return f"No thought found with id: {thought_id}"
    title = thought.title
    full_content = thought.full_content
    
    text = f"### Title:{title}\n\nContent:\n{full_content}"
    return text
