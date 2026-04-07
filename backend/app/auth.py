from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from authlib.integrations.starlette_client import OAuth
from fastapi import Depends, HTTPException, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings, get_settings
from app.db import get_db
from app.models import User


settings = get_settings()
oauth = OAuth()

if settings.google_client_id and settings.google_client_secret:
    oauth.register(
        name="google",
        client_id=settings.google_client_id,
        client_secret=settings.google_client_secret,
        server_metadata_url="https://accounts.google.com/.well-known/openid-configuration",
        client_kwargs={"scope": "openid email profile"},
    )


def google_auth_configured() -> bool:
    return bool(settings.google_client_id and settings.google_client_secret)


def user_is_admin(user: User, app_settings: Settings | None = None) -> bool:
    cfg = app_settings or settings
    return user.email.lower() in cfg.parsed_admin_emails


def create_access_token(user: User, app_settings: Settings | None = None) -> str:
    cfg = app_settings or settings
    expires_at = datetime.now(UTC) + timedelta(days=cfg.auth_token_expiry_days)
    payload = {
        "sub": str(user.id),
        "email": user.email,
        "name": user.full_name,
        "exp": expires_at,
        "iat": datetime.now(UTC),
    }
    return jwt.encode(payload, cfg.jwt_secret_key, algorithm="HS256")


def decode_access_token(token: str, app_settings: Settings | None = None) -> dict[str, Any]:
    cfg = app_settings or settings
    try:
        return jwt.decode(token, cfg.jwt_secret_key, algorithms=["HS256"])
    except jwt.PyJWTError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is invalid or expired.",
        ) from exc


async def upsert_google_user(db: AsyncSession, userinfo: dict[str, Any]) -> User:
    google_sub = str(userinfo["sub"])
    email = userinfo.get("email", "").lower()

    result = await db.execute(select(User).where(User.google_sub == google_sub))
    user = result.scalar_one_or_none()

    if user is None:
        user = User(
            google_sub=google_sub,
            email=email,
            full_name=userinfo.get("name"),
            avatar_url=userinfo.get("picture"),
        )
        db.add(user)
    else:
        user.email = email
        user.full_name = userinfo.get("name")
        user.avatar_url = userinfo.get("picture")
        user.last_login_at = datetime.now(UTC)

    await db.commit()
    await db.refresh(user)
    return user


async def get_current_user(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> User:
    token = request.cookies.get(settings.auth_cookie_name)
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated.")

    payload = decode_access_token(token)
    user_id = int(payload["sub"])
    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found.")
    return user


async def get_current_admin(
    current_user: User = Depends(get_current_user),
) -> User:
    if not user_is_admin(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access is required.")
    return current_user
