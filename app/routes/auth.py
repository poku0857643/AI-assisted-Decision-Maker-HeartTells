from datetime import timedelta, datetime
from fastapi import APIRouter, Depends, HTTPException, status, Response
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.auth.models import Token, UserResponse, RefreshTokenRequset
from app.auth.jwt_handler import create_token_pair, verify_refresh_token