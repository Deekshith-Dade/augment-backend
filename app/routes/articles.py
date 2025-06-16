from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_async_db
from app.utils.ext_articles import scrape_article, to_pgvector
from app.llm_utils.embeddings.embeddings import embed_text_openai
from app.models.models import ExternalAritcle, Thought
from app.llm_utils.tags import generate_article_tags
import numpy as np
from sqlalchemy import select



user_id = "83172f77-5d45-4ec2-ac7e-13e3d0f26504"

router = APIRouter(prefix="/articles", tags=["articles"])

@router.post("/embed")
async def embed_article(payload: dict, db: AsyncSession = Depends(get_async_db)):
    try:
        url = payload["url"]
        article = scrape_article(url)
        
        text = article["text"]
        print(f"Text length: {len(text)}")
        embedding = await embed_text_openai(text)
        tags = generate_article_tags(text)
        
        # Convert timezone-aware datetime to timezone-naive datetime
        published_at = article["published"]
        if published_at and published_at.tzinfo is not None:
            published_at = published_at.replace(tzinfo=None)
            
        new_article = ExternalAritcle(
            url=url,
            title=article["title"],
            embedding=embedding,
            authors=article["authors"],
            text=text,
            excerpt=article["summary"],
            source=article["source_url"],
            top_image_url=article["top_image"],
            published_at=published_at,
            tags=tags,
        )
        db.add(new_article)
        await db.commit()
        return {"url": url, "message": "Article embedded successfully"}
    except Exception as e:
        print(e)
        return {"url": url, "message": "Failed to embed article"}


@router.get("/search")
async def search_articles(query: str, top_k: int = 5, db: AsyncSession = Depends(get_async_db)):
    print(query)
    try:
        query_embedding = await embed_text_openai(query)
        sql = text("""
            SELECT id, title, url, authors, embedding <#> (:embedding)::vector AS distance
            FROM external_articles
            ORDER BY embedding <#> (:embedding)::vector
            LIMIT :top_k
        """)
        result = await db.execute(sql, {"embedding": to_pgvector(query_embedding), "top_k": top_k})
        result = result.all()
        articles = []
        for row in result:
            articles.append({
                "id": row[0],
                "title": row[1],
                "url": row[2],
                "authors": row[3],
                "distance": row[4],
            })
        return {"results": articles}
    except Exception as e:
        print(e)
        return {"query": query, "message": "Failed to search articles"}
    

@router.get("/discover")
async def discover_articles(db: AsyncSession = Depends(get_async_db), limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0)):
    try:
        thoughts = await db.execute(select(Thought.embedding).where(Thought.user_id == user_id))
        embeddings = [row[0] for row in thoughts.fetchall()]
        if not embeddings:
            return []
        user_vector = np.mean(np.array(embeddings), axis=0)
        sql = text("""
            SELECT id, title, text, authors, top_image_url, tags, url, published_at, embedding <#> (:embedding)::vector AS distance
            FROM external_articles
            ORDER BY embedding <#> (:embedding)::vector
            LIMIT :limit
            OFFSET :offset
        """)
        results = await db.execute(sql, {"embedding": to_pgvector(user_vector), "limit": limit, "offset": offset})
        results = results.all()
        articles = []
        for row in results:
            articles.append({
                "id": row[0],
                "title": row[1],
                "excerpt": row[2][:800] if row[2] else "",
                "authors": row[3],
                "imageUrl": row[4],
                "category": row[5][0] if row[5] and len(row[5]) > 0 else None,
                "url": row[6],
                "publishedDate": row[7].strftime("%Y-%m-%d") if row[7] else None,
                "distance": row[8],
            })
        return {"articles": articles}
    except Exception as e:
        print(e)
        return {"message": "Failed to discover articles"}