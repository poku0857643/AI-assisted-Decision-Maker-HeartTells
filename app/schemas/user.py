from pydantic import BaseModel, EmailStr
from typing import Optional

class GoogleUserInfo(BaseModel):
    google_id: str
    email: EmailStr
    name: str
    picture: Optional[str] = None
    email_verified: bool = False

class UserResponse(BaseModel):
    email: EmailStr
    name: str
    picture: Optional[str] = None

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    user: GoogleUserInfo