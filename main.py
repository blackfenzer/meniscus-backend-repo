from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import protected , auth , auth2 , validate_handler
from app.database.session import Base, engine
from app.security.csrf_handler import CsrfProtect, csrf_protect_exception_handler
from fastapi_csrf_protect.exceptions import CsrfProtectError

# Create tables (only for development)
Base.metadata.create_all(bind=engine)

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CSRF Protection
app.add_exception_handler(CsrfProtectError, csrf_protect_exception_handler)

# Include Routes
# app.include_router(auth.router)
app.include_router(protected.router)
app.include_router(auth.router)
app.include_router(auth2.router , prefix="/api/v1" ,tags=["auth2"])
app.include_router(validate_handler.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI with JWT, CSRF, and SQLAlchemy"}
