from fastapi_csrf_protect import CsrfProtect
from fastapi_csrf_protect.exceptions import CsrfProtectError
from fastapi import HTTPException, Request
import os


@CsrfProtect.load_config
def get_csrf_config():
    return [("secret_key", os.getenv("CSRF_SECRET_KEY"))]


def csrf_protect_exception_handler(request: Request, exc: CsrfProtectError):
    raise HTTPException(status_code=exc.status_code, detail=exc.message)
