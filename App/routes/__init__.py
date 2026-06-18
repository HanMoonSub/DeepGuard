from fastapi import APIRouter
from routes.auth import router as auth_router    
from routes.explain import router as explain_router    
from routes.home import router as home_router  
from routes.image import router as image_router  
from routes.inference import router as inference_router  
from routes.video import router as video_router  

api_router = APIRouter(prefix="/api", tags=["api"])
api_router.include_router(auth_router)   
api_router.include_router(explain_router) 
api_router.include_router(home_router)  
api_router.include_router(image_router)
api_router.include_router(inference_router) 
api_router.include_router(video_router)        