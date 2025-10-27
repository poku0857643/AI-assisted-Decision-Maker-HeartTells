from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.google_oauth import router as google_oauth_router
from app.config.config import settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

app.include_router(google_oauth_router, prefix="/auth/google", tags=["authentication"])


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
