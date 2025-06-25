from sqlalchemy import text

async def get_similar_thoughts(db, query_embedding, user_id, top_k=5):
    sql = text("""
        SELECT id, title, full_content, embedding <#> (:embedding)::vector AS distance, created_at
        FROM thoughts
        WHERE user_id = :user_id
        ORDER BY embedding <#> (:embedding)::vector, created_at DESC
        LIMIT :top_k
    """)
    result = await db.execute(
        sql,
        {
            "embedding": query_embedding,
            "user_id": user_id,
            "top_k": top_k
        }
    )
    keyed_results = []
    for r in result.all():
        keyed_results.append({
            "id": r[0],
            "title": r[1],
            "full_content": r[2],
            "distance": r[3],
            "created_at": r[4]
        })
    return keyed_results


def to_pgvector(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


def pretty_thoughts(results):
    texts = []
    for result in results:
        id, title, full_content, distance, created_at = result['id'], result['title'], result['full_content'], result['distance'], result['created_at']
        texts.append(f"### ID:{id} \n\n ### Title:{title}({-distance:.2f})\n\nContent:\n{full_content[:300]}")
    final_text = "\n\n".join(texts)
    return final_text