"""Cookie session authentication for the Scribe API."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, EmailStr, Field, field_validator
import bcrypt

from database.repositories import create_user, get_user_by_email, get_user_by_id

router = APIRouter(prefix="/api/auth", tags=["auth"])


class Credentials(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1)

    @field_validator("email", mode="before")
    @classmethod
    def normalize_email(cls, value: object) -> str:
        return str(value or "").strip().lower()


class Registration(Credentials):
    name: str = Field(min_length=1, max_length=160)

    @field_validator("name", mode="before")
    @classmethod
    def normalize_name(cls, value: object) -> str:
        return str(value or "").strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str) -> str:
        if len(value) < 6:
            raise ValueError("Password must be at least 6 characters.")
        return value


def public_user(user: object) -> dict[str, object]:
    return {"id": getattr(user, "id"), "name": getattr(user, "name"), "email": getattr(user, "email")}


def get_current_user(request: Request) -> dict[str, object]:
    user_id = request.session.get("user_id")
    if not isinstance(user_id, int):
        raise HTTPException(status_code=401, detail="Authentication required.")
    user = get_user_by_id(user_id)
    if user is None:
        request.session.clear()
        raise HTTPException(status_code=401, detail="Authentication required.")
    return public_user(user)


@router.post("/register")
def register(payload: Registration):
    email = payload.email.strip().lower()
    if get_user_by_email(email):
        raise HTTPException(status_code=409, detail="Email already registered.")
    password_hash = bcrypt.hashpw(payload.password.encode("utf-8"), bcrypt.gensalt()).decode("ascii")
    try:
        user = create_user(payload.name, email, password_hash)
    except Exception as exc:
        # A concurrent registration can pass the first lookup; don't leak DB details.
        from sqlalchemy.exc import IntegrityError
        if isinstance(exc, IntegrityError):
            raise HTTPException(status_code=409, detail="Email already registered.") from exc
        raise
    return public_user(user)


@router.post("/login")
def login(payload: Credentials, request: Request):
    user = get_user_by_email(payload.email.strip().lower())
    valid = False
    if user is not None:
        try:
            valid = bcrypt.checkpw(payload.password.encode("utf-8"), user.password_hash.encode("ascii"))
        except (ValueError, TypeError):
            valid = False
    if not valid:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    request.session.clear()
    request.session["user_id"] = int(user.id)
    return public_user(user)


@router.get("/me")
def me(user: dict[str, object] = Depends(get_current_user)):
    return user


@router.post("/logout")
def logout(request: Request, _user: dict[str, object] = Depends(get_current_user)):
    request.session.clear()
    return Response(status_code=204)
