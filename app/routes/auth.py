from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.auth.models import Token, UserResponse, RefreshTokenRequset, RefreshTokenRequest
from app.auth.jwt_handler import create_token_pair, verify_refresh_token
from app.database.connection import get_db
from app.database.crud import UserCRUD, RefreshTokenCRUD
from app.auth.dependencies import get_current_active_user
from app.database.models import User
from app.config import settings

router = APIRouter()

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user: UserCreate, db: Session = Depends(get_db)):
    user_crud = UserCRUD(db)

    # Check if user already exists
    if user_crud.get_user_by_username(user.username):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "User already exists"
        )

    if user_crud.get_user_by_email(user.email):
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail = "Email already exists"
        )

    # Create new user
    db_user = user_crud.create_user(
        username = user.username,
        email = user.email,
        full_name=user.full_name,
        password = user.password,
        role=user.role
    )
    return db_user

@router.post("/login", response_model=Token, status_code=status.HTTP_200_OK)
async def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user_crud = UserCRUD(db)
    user = user_crud.get_user_by_email(form_data.username)

    # Authenticate user
    user = user_crud.authenticate_user(login.username, login_data.password)
    if not user:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail = "Invalid credentials",
            headers= {"WWW-Authenticate": "Bearer"}
        )

    # Update last login
    user_crud.update_last_login(user.id)

    # Create token pair
    token_data = create_token_pair(user.id, user.username, user.role)


    # store refresh token in database
    refresh_crud.create_refresh_token(
        user_id=user.id,
        token=token_data["refresh_token"],
        expires_at=datetime.utcnow() + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    )

@router.post("/login-form", response_model=Token)
async def login_with_form(
        form_data: OAuth2PasswordRequestForm = Depends(),
        db: Session = Depends(get_db)
):
    user_crud = UserCRUD(db)
    refresh_crud = RefreshTokenRequset(db)

    # Authenticate user
    user = user_crud.authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Bearer"}
        )

    # Update last login
    user_crud.update_last_login(user.id)

    # Create token pair
    token_data = create_token_pair(user.id, user.username, user.role)

    # Store refresh token in database
    refresh_crud.create_refresh_token(
        user_id=user.id,
        token=token_data["refresh_token"],
        expires_at=datetime.utcnow() + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    )
    return token_data

@router.post("/refresh-token", response_model=Token)
async def refresh_token(
        token_request: RefreshTokenRequset = Depends(),
        db: Session = Depends(get_db)
):
    refresh_crud = RefreshTokenCRUD(db)
    user_crud = UserCRUD(db)

    # verify refresh token
    payload = verify_refresh_token(token_request.refresh_token)
    if not payload:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Check if token exists in database and is valid
    db_token = refresh_crud.get_refresh_token(token_request.refresh_token)
    if not db_token:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Get User
    user = user_crud.get_user_by_id(db_token.user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code = status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )

    # Revoke old refresh token
    refresh_crud.revoke_token(token_request.refresh_token)

    # Create new token pair
    token_data = create_token_pair(user.id, user.username, user.role)

    # Store new refresh token
    refresh_crud.create_refresh_token(
        user_id = user.id,
        token = token_data["refresh_token"],
        expires_at = datetime.utcnow() + timedelta(minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES)
    )

    return token_data

@router.post("/logout")
async def logout(
        token_request: RefreshTokenRequest,
        current_user: User = Depends(get_current_active_user),
        db: Session = Depends(get_db)
):

    # Revoke refresh token
    refresh_crud = RefreshTokenCRUD(token_request.refresh_token)
    return {"message": "Successfully logged out"}

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user: User = Depends(get_current_active_user)):
    return current_user
