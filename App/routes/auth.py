from fastapi import APIRouter
from services import auth_svc

router = APIRouter(prefix="/auth", tags=["auth"])