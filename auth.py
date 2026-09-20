"""
auth.py — JWT + bcrypt + auth dependencies
==========================================
Security:
  - bcrypt password hashing (passlib)
  - JWT tokens (7-day expiry) — HS256, secret from env
  - get_current_user dependency (Bearer token)
  - require_admin (hardcoded creds, env-overridable: admin / admin123)
  - login rate limit helper (5 fails / 15 min per ip+email)
"""
import os
import time
from datetime import datetime, timedelta, timezone

from fastapi import Depends, Header, HTTPException, Request
from passlib.context import CryptContext
import jwt  # pyjwt

from database import get_db, User
from sqlalchemy.orm import Session

SECRET_KEY = os.getenv("JWT_SECRET", "medai-fyp-secret-change-in-production-2026")
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 7

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

_login_fails: dict[tuple, list] = {}
MAX_LOGIN_FAILS = 5
LOGIN_WINDOW_S = 15 * 60


def hash_password(pw: str) -> str:
    return pwd_context.hash(pw)


def verify_password(pw: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(pw, hashed)
    except Exception:
        return False


def create_token(user_id: int, email: str) -> str:
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(days=TOKEN_EXPIRE_DAYS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        return None


def register_login_fail(ip: str, email: str) -> bool:
    key = (ip, email.lower())
    now = time.time()
    fails = [t for t in _login_fails.get(key, []) if now - t < LOGIN_WINDOW_S]
    if len(fails) >= MAX_LOGIN_FAILS:
        _login_fails[key] = fails
        return False
    _login_fails[key] = fails
    return True


def record_login_fail(ip: str, email: str):
    _login_fails.setdefault((ip, email.lower()), []).append(time.time())


def clear_login_fails(ip: str, email: str):
    _login_fails.pop((ip, email.lower()), None)


def get_client_ip(request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def get_current_user(
    request: Request = None,
    db: Session | None = Depends(get_db),
    authorization: str = Header(default=""),
) -> User:
    if db is None:
        raise HTTPException(503, "Database not configured — set DATABASE_URL env.")

    token = ""
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    elif request is not None:
        token = request.cookies.get("medai_token", "")

    payload = decode_token(token)
    if not payload:
        raise HTTPException(401, "Session expired or invalid — please log in again.")

    user = db.query(User).filter(User.id == int(payload.get("sub", 0))).first()
    if not user or not user.is_active:
        raise HTTPException(401, "Account not found or deactivated.")
    return user


def require_admin(username: str = "", password: str = "") -> bool:
    return username == ADMIN_USERNAME and password == ADMIN_PASSWORD
