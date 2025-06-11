from sqlalchemy.orm import Session
from app.models.models import Thought, Tag
from slugify import slugify


def assign_tags_to_thought(db: Session, user_id: str, thought_id: str, tag_names: list[str]):
    thought = db.query(Thought).filter_by(id=thought_id, user_id=user_id).first()
    if not thought:
        raise ValueError(f"Thought with id {thought_id} not found")
    
    tags = []
    for name in tag_names:
        tag = db.query(Tag).filter_by(name=name, user_id=user_id).first()
        if not tag:
            tag = Tag(name=name, slug=slugify(name), user_id=user_id)
            db.add(tag)
            db.flush()
        tags.append(tag)

    thought.tags = tags
    db.commit()
