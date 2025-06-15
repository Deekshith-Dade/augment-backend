from newspaper import Article


def scrape_article(url: str):
    article = Article(url)
    article.download()
    article.parse()

    return {
        "title": article.title,
        "text": article.text,
        "authors": article.authors,
        "published": article.publish_date,
        "top_image": article.top_image,
        "source_url": article.source_url,
        "images": article.images,
        "videos": article.movies,
        "keywords": article.keywords,
        "summary": article.summary,
        "html": article.html,
    }
    
def to_pgvector(vec):
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"