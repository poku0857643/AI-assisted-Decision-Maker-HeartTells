from fastapi import APIRouter, HTTPException, Request, Depends, Header
from fastapi.responses import RedirectResponse
from google.auth.transport.requests import Request as GoogleRequest
from google.oauth2 import id_token
from google_auth_oauthlib.flow import Flow
import json
from jose import jwt, JWTError
from datetime import datetime, timedelta
from typing import Optional
from app.config.config import settings
from app.schemas.user import GoogleUserInfo, TokenResponse, UserResponse

router = APIRouter()

SCOPES = ['openid', 'email', 'profile']

def create_google_flow():
    """Create Google OAuth flow"""
    flow = Flow.from_client_config(
        {
            "web": {
                "client_id": settings.GOOGLE_CLIENT_ID,
                "client_secret": settings.GOOGLE_CLIENT_SECRET,
                "auth_uri": "https://accounts.google.com/o/oauth2/auth",
                "token_uri": "https://oauth2.googleapis.com/token",
                "redirect_uris": [settings.GOOGLE_REDIRECT_URI]
            }
        },
        scopes=SCOPES
    )
    flow.redirect_uri = settings.GOOGLE_REDIRECT_URI
    return flow

def create_access_token(data: dict):
    """Create JWT access token"""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    return encoded_jwt

@router.get("/login")
async def google_login():
    """Initiate Google OAuth login"""
    try:
        flow = create_google_flow()
        authorization_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true'
        )
        return {"authorization_url": authorization_url, "state": state}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to initiate Google OAuth: {str(e)}")

@router.get("/callback", response_model=TokenResponse)
async def google_callback(request: Request):
    """Handle Google OAuth callback"""
    try:
        flow = create_google_flow()
        
        # Get the authorization response from the callback URL
        authorization_response = str(request.url)
        flow.fetch_token(authorization_response=authorization_response)
        
        # Get user info from Google
        credentials = flow.credentials
        request_session = GoogleRequest()
        
        # Verify the ID token
        idinfo = id_token.verify_oauth2_token(
            credentials.id_token, 
            request_session, 
            settings.GOOGLE_CLIENT_ID
        )
        
        # Extract user information
        user_data = {
            "google_id": idinfo.get("sub"),
            "email": idinfo.get("email"),
            "name": idinfo.get("name"),
            "picture": idinfo.get("picture"),
            "email_verified": idinfo.get("email_verified", False)
        }
        
        # Create JWT token
        access_token = create_access_token(data={"sub": user_data["email"]})
        
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user": user_data
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process Google OAuth callback: {str(e)}")

def get_token_from_header(authorization: Optional[str] = Header(None)):
    """Extract JWT token from Authorization header"""
    if authorization is None:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    try:
        scheme, token = authorization.split()
        if scheme.lower() != "bearer":
            raise HTTPException(status_code=401, detail="Invalid authentication scheme")
        return token
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid authorization header format")

@router.get("/user", response_model=UserResponse)
async def get_current_user(token: str = Depends(get_token_from_header)):
    """Get current user information from JWT token"""
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid authentication credentials")
        return {"email": email, "name": "", "picture": None}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

@router.get("/callback/frontend")
async def google_callback_frontend(request: Request, frontend_url: str = "http://localhost:3000"):
    """Handle Google OAuth callback and redirect to frontend with token"""
    try:
        flow = create_google_flow()
        
        authorization_response = str(request.url)
        flow.fetch_token(authorization_response=authorization_response)
        
        credentials = flow.credentials
        request_session = GoogleRequest()
        
        idinfo = id_token.verify_oauth2_token(
            credentials.id_token, 
            request_session, 
            settings.GOOGLE_CLIENT_ID
        )
        
        user_data = {
            "google_id": idinfo.get("sub"),
            "email": idinfo.get("email"),
            "name": idinfo.get("name"),
            "picture": idinfo.get("picture"),
            "email_verified": idinfo.get("email_verified", False)
        }
        
        access_token = create_access_token(data={"sub": user_data["email"]})
        
        # Redirect to frontend with token and user data
        redirect_url = f"{frontend_url}/auth/callback?token={access_token}&email={user_data['email']}&name={user_data['name']}"
        return RedirectResponse(url=redirect_url)
        
    except Exception as e:
        error_url = f"{frontend_url}/auth/error?message={str(e)}"
        return RedirectResponse(url=error_url)