from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from app.database.database import get_async_db
from app.utils.ext_articles import scrape_article, to_pgvector
from app.llm_utils.embeddings.embeddings import embed_text_openai
from app.models.models import ExternalAritcle

router = APIRouter(prefix="/articles", tags=["articles"])

@router.post("/embed")
async def embed_article(payload: dict, db: AsyncSession = Depends(get_async_db)):
    try:
        url = payload["url"]
        article = scrape_article(url)
        
        text = article["text"]
        print(f"Text length: {len(text)}")
        embedding = await embed_text_openai(text)
        
        new_article = ExternalAritcle(
            url=url,
            title=article["title"],
            embedding=embedding,
            authors=article["authors"],
            text=text,
            excerpt=article["summary"],
            source=article["source_url"],
            top_image_url=article["top_image"],
            published_at=article["published"],
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