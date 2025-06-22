from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_async_db
from app.utils.ext_articles import scrape_article, to_pgvector
from app.llm_utils.embeddings.embeddings import embed_text_openai
from app.models.models import ExternalAritcle, Thought
from app.llm_utils.tags import generate_article_tags
import numpy as np
from sqlalchemy import select
from app.routes.utils import get_current_user
from app.models.models import User
from app.core.logging import logger
from app.core.limiter import limiter, rate_limits

router = APIRouter(prefix="/articles", tags=["articles"])

@router.post("/embed")
async def embed_article(request: Request, payload: dict, db: AsyncSession = Depends(get_async_db)):
    logger.info("embed_article_request", url=payload["url"])
    try:
        url = payload["url"]
        article = scrape_article(url)
        logger.info("embed_article_request_article", url=url)
        text = article["text"]
        embedding = await embed_text_openai(text)
        logger.info("embed_article_request_embedding", url=url, len_embedding=len(embedding))
        tags = generate_article_tags(text)
        logger.info("embed_article_request_tags", url=url, tags=tags)
        
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
        logger.info("embed_article_request_return", url=url)
        return {"url": url, "message": "Article embedded successfully"}
    except Exception as e:
        logger.error("embed_article_request_error", url=url, error=str(e))
        return {"url": url, "message": "Failed to embed article"}


@router.get("/search")
async def search_articles(request: Request, query: str, top_k: int = 5, db: AsyncSession = Depends(get_async_db), user: User = Depends(get_current_user)):
    logger.info("search_articles_request", query=query, top_k=top_k)
    try:
        query_embedding = await embed_text_openai(query)
        logger.info("search_articles_request_query_embedding", query=query, len_embedding=len(query_embedding))
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
        logger.info("search_articles_request_return", query=query, top_k=top_k, len_articles=len(articles))
        return {"results": articles}
    except Exception as e:
        logger.error("search_articles_request_error", query=query, error=str(e))
        return {"query": query, "message": "Failed to search articles"}
    

@router.get("/discover")
@limiter.limit(rate_limits["RATE_LIMIT_DISCOVER_ARTICLES"][0])
async def discover_articles(request: Request, db: AsyncSession = Depends(get_async_db), limit: int = Query(10, ge=1, le=50),
    offset: int = Query(0, ge=0), user: User = Depends(get_current_user)):
    logger.info("discover_articles_request", limit=limit, offset=offset)
    try:
        thoughts = await db.execute(select(Thought.embedding).where(Thought.user_id == user.id))
        embeddings = [row[0] for row in thoughts.fetchall()]
        if not embeddings:
            logger.info("discover_articles_request_no_embeddings", limit=limit, offset=offset)
            return {"articles": []}
        user_vector = np.mean(np.array(embeddings), axis=0)
        logger.info("discover_articles_request_user_vector", limit=limit, offset=offset, len_user_vector=len(user_vector))
        sql = text("""
            SELECT id, title, text, authors, top_image_url, tags, url, published_at, embedding <#> (:embedding)::vector AS distance
            FROM external_articles
            ORDER BY embedding <#> (:embedding)::vector
            LIMIT :limit
            OFFSET :offset
        """)
        results = await db.execute(sql, {"embedding": to_pgvector(user_vector), "limit": limit, "offset": offset})
        results = results.all()
        logger.info("discover_articles_request_results", limit=limit, offset=offset, len_results=len(results))
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
        logger.info("discover_articles_request_return", limit=limit, offset=offset, len_articles=len(articles))
        return {"articles": articles}
    except Exception as e:
        logger.error("discover_articles_request_error", limit=limit, offset=offset, error=str(e))
        return {"message": "Failed to discover articles"}