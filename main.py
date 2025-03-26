import os
import time
import sys
import uuid
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from app.routes import (
    auth2,
    validate_xgboost,
    validate,
    torch,
    crud_CSVfile,
    crud_model,
    crud_user,
)
from app.database.session import Base, engine
from app.security.csrf_handler import CsrfProtect, csrf_protect_exception_handler
from fastapi_csrf_protect.exceptions import CsrfProtectError
from loguru import logger
from dotenv import load_dotenv
import uuid
import traceback
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

load_dotenv()
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_PATH = os.getenv("LOG_PATH", "logs/app.log")
# Create tables (only for development)
Base.metadata.create_all(bind=engine)

app = FastAPI(debug=True)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://34.226.203.199:3010",
        "http://localhost:3010",
        "http://localhost:3000",
        "http://nextjs:3000",
        "http://nextjs:3010",
        "http://localhost:8000",
        "http://fastapi:8000",
        "http://34.226.203.199"
    ],  # Allow Next.js frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# CSRF Protection
app.add_exception_handler(CsrfProtectError, csrf_protect_exception_handler)

logger.configure(
    handlers=[
        {
            "sink": sys.stdout,
            "format": "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<magenta>{extra[request_id]}</magenta> - "
            "<level>{message}</level>",
            "colorize": True,
        },
        # For production
        # {
        #     "sink": "logs/app.json.log",
        #     "serialize": True,  # Write JSON
        #     "rotation": "500 MB",
        #     "compression": "zip",
        #     "retention": "30 days",
        # },
        # {
        #     "sink": "logs/app.errors.log",
        #     "level": "WARNING",
        #     "rotation": "00:00",
        #     "retention": "90 days",
        # },
    ]
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Create a unique error ID
    error_id = str(uuid.uuid4())

    # Get the full traceback
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

    # Log the detailed error with the unique ID and traceback
    logger.opt(depth=1).error(f"Unhandled exception [{error_id}]: {exc}")
    logger.opt(depth=1).error(f"Traceback [{error_id}]:\n{tb_str}")

    # In development, you can return the actual error details
    if app.debug:
        return JSONResponse(
            status_code=500,
            content={"error_id": error_id, "message": str(exc), "traceback": tb_str},
        )
    # In production, return a clean error with reference ID
    else:
        return JSONResponse(
            status_code=500,
            content={
                "error_id": error_id,
                "message": "Internal Server Error. Reference ID for support: "
                + error_id,
            },
        )


@app.middleware("http")
async def logging_middleware(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    client_ip = request.client.host if request.client else "unknown"

    # Add contextual information for all logs during this request
    with logger.contextualize(request_id=request_id):
        logger.info(
            "Request started",
            method=request.method,
            path=request.url.path,
            client_ip=client_ip,
        )

        start_time = time.time()
        response = None

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request failed: {str(e)}")
            raise
        finally:
            process_time = time.time() - start_time
            status_code = response.status_code if response else 500

            logger.info(
                "Request completed",
                method=request.method,
                path=request.url.path,
                status_code=status_code,
                process_time=f"{process_time:.4f}s",
            )

        response.headers["X-Request-ID"] = request_id
        return response


# Include Routes
app.include_router(auth2.router, prefix="/api/v1", tags=["auth2"])
app.include_router(validate.router, prefix="/api/v1", tags=["validate"])
app.include_router(validate_xgboost.router, prefix="/api/v1", tags=["validate_xg_boost"])
app.include_router(torch.router, prefix="/api/v1/nn", tags={"test"})
app.include_router(crud_CSVfile.router, prefix="/api/v1/csv_files", tags=["CSV file"])
app.include_router(crud_model.router, prefix="/api/v1/model", tags=["model"])
app.include_router(crud_user.router, prefix="/api/v1/users", tags=["user"])


@app.get("/")
def read_root():
    return {"message": "Welcome to FastAPI with JWT, CSRF, and SQLAlchemy"}
