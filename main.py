from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.routes.google_oauth import router as google_oauth_router
from app.routes.ecg import router as ecg_router
from app.routes.decision_analysis import router as decision_router
from app.config.config import settings

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(google_oauth_router, prefix="/auth/google", tags=["authentication"])
app.include_router(ecg_router, prefix="/api/v1", tags=["ECG"])
app.include_router(decision_router, prefix="/api/v1", tags=["Decision Analysis"])


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/login")
async def login_page():
    return FileResponse("templates/login.html")

@app.get("/dashboard")
async def dashboard_page():
    return {"message": "Dashboard - TODO: Create dashboard page"}

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}
