from fastapi import APIRouter
from api_routes.inference import router as inference_router

router = APIRouter()
router.include_router(inference_router)