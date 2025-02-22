from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import (
    auth2,
    validate,
    clean,
    torch,
    crud_CSVfile,
    crud_model,
    crud_user,
)
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
app.include_router(auth2.router, prefix="/api/v1", tags=["auth2"])
app.include_router(validate.router, prefix="/smart", tags=["smart"])
app.include_router(clean.router, prefix="/smart", tags={"smart"})
app.include_router(torch.router, prefix="/nn", tags={"test"})
app.include_router(crud_CSVfile.router, prefix="/api/v1/csv_files", tags=["CSV file"])
app.include_router(crud_model.router, prefix="/api/v1/model", tags=["model"])
app.include_router(crud_user.router, prefix="/api/v1/users", tags=["user"])
# app.include_router(nn_model.router, prefix="/api/v1/model", tags=["torch"])


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI with JWT, CSRF, and SQLAlchemy"}
