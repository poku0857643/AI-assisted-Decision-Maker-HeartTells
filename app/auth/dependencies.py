from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from typing import List, Optional

from app.auth.jwt_handler import verify_access_token
from app.database.crud import UserCRUD
from app.database.models import User, UserRole
from app.database.connection import get_db

# Security scheme
security  = HTTPBearer()

async def get_current_user(
        credentials: HTTPAuthorizationCredentials = Depends(security),
        db: Session = Depends(get_db)
) -> User:
    token = credentials.credentials

    # verify token
    payload = verify_access_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    username: str = payload.get("sub")
    user_id : int = payload.get("user_id")

    if username is None or user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Get user from database
    user_crud = UserCRUD(db)
    user = user_crud.get_user_by_id(user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )


    if not user.is_active:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "User is not active",
            headers = {"WWW-Authenticate": "Bearer"}
        )

    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> UserRole:
    if not current_user.is_active:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "User is not active",
        )
    return current_user

def require_roles(allowed_roles: List[UserRole]):
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.rolw not in allowed_roles:
            raise HTTPException(
                status_code = status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions"
            )
        return current_user
    return role_checker

async def get_current_user_optional(
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
        db: Session = Depends(get_db)
) -> Optional[User]:

    if not credentials:
        return None

    try:
        token = credentials.credentials

        # verify token
        payload = verify_access_token(token)
        if not payload:
            return None

        username: str = payload.get("sub")
        user_id : int = payload.get("user_id")

        if username is None or user_id is None:
            return None

        # Get user from database
        user_crud = UserCRUD(db)
        user = user_crud.get_user_by_id(user_id)

        if user is None or not user.is_active:
            return None

    except Exception:
        return None
