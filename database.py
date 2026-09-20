"""
database.py — Neon Postgres (SQLAlchemy)
========================================
Saara user data yahan rehta hai: users, analyses (history), chat sessions,
chat messages, aur per-user assistant memory.

Env:
  DATABASE_URL  Neon connection string (pooled), e.g.
                postgresql+psycopg://user:pass@ep-xxx.aws.neon.tech/neondb?sslmode=require
"""
import os
from datetime import datetime, timezone

from sqlalchemy import (
    create_engine, Column, Integer, String, Text, Float, Boolean,
    DateTime, ForeignKey, Index,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "")

# Neon ka raw string `postgresql://...` hota hai — SQLAlchemy ko driver bhi chahiye
# (psycopg3). Dono schemes accept karo taake user ka paste kiya hua URL seedha chale.
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

if not DATABASE_URL:
    _engine = None
    SessionLocal = None
else:
    _engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
        pool_recycle=280,
        pool_size=3,
        max_overflow=5,
        connect_args={"connect_timeout": 10} if DATABASE_URL.startswith("postgres") else {},
    )
    SessionLocal = sessionmaker(bind=_engine, autoflush=False, expire_on_commit=False)

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    full_name = Column(String(120), nullable=False, default="")
    age = Column(String(10), default="")
    gender = Column(String(20), default="")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    is_active = Column(Boolean, default=True)

    analyses = relationship("Analysis", back_populates="user", cascade="all, delete-orphan")
    chat_sessions = relationship("ChatSession", back_populates="user", cascade="all, delete-orphan")
    memories = relationship("UserMemory", back_populates="user", cascade="all, delete-orphan")


class Analysis(Base):
    """Har predict + PDF ek row — profile history isi se banti hai."""
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    model_type = Column(String(20), nullable=False, index=True)
    scan_type = Column(String(40), default="")
    result = Column(String(80), nullable=False)
    confidence = Column(Float, nullable=False)
    inconclusive = Column(Boolean, default=False)
    scan_type_warning = Column(Text, default="")
    result_json = Column(Text, default="{}")
    thumbnail_b64 = Column(Text, default="")
    patient_name = Column(String(120), default="")
    patient_age = Column(String(10), default="")
    patient_gender = Column(String(20), default="")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False, index=True)

    user = relationship("User", back_populates="analyses")


class ChatSession(Base):
    __tablename__ = "chat_sessions"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    title = Column(String(120), default="New chat")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="chat_sessions")
    messages = relationship("ChatMessage", back_populates="session", cascade="all, delete-orphan",
                            order_by="ChatMessage.id")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id = Column(Integer, primary_key=True)
    session_id = Column(Integer, ForeignKey("chat_sessions.id"), nullable=False, index=True)
    role = Column(String(12), nullable=False)
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="messages")


class UserMemory(Base):
    """Assistant ki per-user knowledge — jo bhi user batata hai, facts yahan save hote hain."""
    __tablename__ = "user_memory"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    key = Column(String(80), nullable=False)
    value = Column(Text, nullable=False)
    source = Column(String(20), default="chat")
    created_at = Column(DateTime(timezone=True), default=utcnow, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False)

    user = relationship("User", back_populates="memories")

    __table_args__ = (
        Index("ix_user_memory_user_key", "user_id", "key"),
    )


def init_db():
    if _engine is None:
        return False
    Base.metadata.create_all(_engine)
    return True


def get_db():
    if SessionLocal is None:
        yield None
        return
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
