from sqlalchemy import Column, String, DateTime, Text, ForeignKey, UniqueConstraint, ARRAY
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, TIMESTAMP
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from app.database.database import Base
from datetime import datetime, timezone
import uuid
from sqlalchemy.orm import relationship

class User(Base):
    __tablename__ = "users"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    external_id = Column(String, nullable=True, unique=True, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=True)
    first_name = Column(String, nullable=True)
    last_name = Column(String, nullable=True)
    profile_image_url = Column(String, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.now(timezone.utc))
    
    thoughts = relationship("Thought", back_populates="user")
    tags = relationship("Tag", back_populates="user")
    chat_sessions = relationship("ChatSession", back_populates="user")

class Tag(Base):
    __tablename__ = "tags"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    name = Column(String, nullable=False, index=True)
    slug = Column(String, nullable=False, index=True)
    __table_args__ = (UniqueConstraint(user_id, name, name="uix_user_id_name"),)
    created_at = Column(DateTime, default=datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="tags")
    thoughts = relationship("Thought", secondary="thought_tags", back_populates="tags")

class Thought(Base):
    __tablename__ = "thoughts"
    
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    title = Column(String, nullable=True)
    text_content = Column(Text, nullable=False)
    image_url = Column(String, nullable=True)
    audio_url = Column(String, nullable=True)
    embedding = Column(Vector(1536), nullable=False)
    full_content = Column(Text, nullable=False)
    meta = Column(JSONB, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="thoughts")
    tags = relationship("Tag", secondary="thought_tags", back_populates="thoughts")

class ThoughtTag(Base):
    __tablename__ = "thought_tags"
    thought_id = Column(PG_UUID(as_uuid=True), ForeignKey("thoughts.id"), primary_key=True)
    tag_id = Column(PG_UUID(as_uuid=True), ForeignKey("tags.id"), primary_key=True)

class ChatSession(Base):
    __tablename__ = "chat_sessions"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    user_id = Column(PG_UUID(as_uuid=True), ForeignKey("users.id"), index=True, nullable=False)
    title = Column(String, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.now(timezone.utc))
    updated_at = Column(TIMESTAMP(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))
    
    user = relationship("User", back_populates="chat_sessions")

class ExternalAritcle(Base):
    __tablename__ = "external_articles"
    id = Column(PG_UUID(as_uuid=True), primary_key=True, default=lambda: uuid.uuid4())
    url = Column(String, unique=True, index=True, nullable=False)
    title = Column(String, nullable=True)
    embedding = Column(Vector(1536), nullable=False)
    authors = Column(ARRAY(String), nullable=True)
    text = Column(Text, nullable=False)
    tags = Column(ARRAY(String), nullable=True)
    excerpt = Column(String, nullable=True)
    source = Column(String, nullable=True)
    top_image_url = Column(String, nullable=True)
    published_at = Column(DateTime, nullable=True)
    created_at = Column(TIMESTAMP(timezone=True), default=datetime.now(timezone.utc), onupdate=datetime.now(timezone.utc))