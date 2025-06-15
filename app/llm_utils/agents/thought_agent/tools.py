import sys
sys.path.append("..")

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, tool
from sqlalchemy import text, select
from app.llm_utils.embeddings.embeddings import embed_text_openai
from app.models.models import Thought
from app.utils.ext_articles import to_pgvector

from dotenv import load_dotenv
load_dotenv()



def pretty_thoughts(results):
    texts = []
    for result in results:
        id, title, full_content, distance = result
        texts.append(f"### ID:{id} \n\n ### Title:{title}({-distance:.2f})\n\nContent:\n{full_content[:100]}")
    final_text = "\n\n".join(texts)
    return final_text


async def get_similar_thoughts(query_embedding, user_id, top_k=5, session=None):
    try:
        sql = text("""
            SELECT id, title, full_content, embedding <#> (:embedding)::vector AS distance
            FROM thoughts
            WHERE user_id = :user_id
            ORDER BY embedding <#> (:embedding)::vector
            LIMIT :top_k
        """)
        result = await session.execute(
            sql,
            {
                "embedding": query_embedding,
                "user_id": user_id,
                "top_k": top_k
            }
        )
        
        return result.all()
    except Exception as e:
        print(e)
        return []





@tool
async def fetch_relevant_thoughts(query: str, config: RunnableConfig) -> str:
    """A Rag Tool that based on a "descriptive" query provided fetches relevant thoughts that the user has posted"""
    print(f"fetch_relevant_thoughts: {query}")
    try:
        user_id = config.get("configurable").get("user_id")
        session = config.get("configurable").get("session")
        print(f"user_id: {user_id}")
        print(f"session: {session}")
    except Exception as e:
        raise("User ID not provided")
    embedding =  await embed_text_openai(query)
    print(f"embedding: {len(embedding)}")
    pg_vector_str = to_pgvector(embedding)
    print(f"pg_vector_str: {len(pg_vector_str)}")
    response = await get_similar_thoughts(pg_vector_str, user_id, top_k=5, session=session)
    print(f"response: {len(response)}")
    result = pretty_thoughts(response)
    return result


@tool
async def get_thought_details(thought_id: str, config: RunnableConfig) -> str:
    """A tool that fetches the information about particular thought requested"""
    try:
        user_id = config.get("configurable").get("user_id")
        session = config.get("configurable").get("session")
    except Exception as e:
        print(f"Tool: get_thought_details: {e}")
        raise("User ID not provided")
    
    
    try:
        stmt = select(Thought).where(Thought.id == thought_id, Thought.user_id == user_id)
        thought = await session.execute(stmt)
        thought = thought.scalars().first()
        if not thought:
            return f"No thought found with id: {thought_id}"
        title = thought.title
        full_content = thought.full_content
        
        text = f"### Title:{title}\n\nContent:\n{full_content}"
        return text
    except Exception as e:
        print(f"Tool: get_thought_details: {e}")
        return f"No thought found with id: {thought_id}"
