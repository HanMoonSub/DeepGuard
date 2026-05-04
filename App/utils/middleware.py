from fastapi import Request, FastAPI
from starlette.middleware.base import BaseHTTPMiddleware
import redis.asyncio as aioredis
import uuid
import json
import logging
import os
from dotenv import load_dotenv

logging.basicConfig(level=logging.CRITICAL)
load_dotenv()

REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", 6379)) 

# Redis setup
redis_pool = aioredis.ConnectionPool(host=REDIS_HOST, port=REDIS_PORT, db=0, max_connections=10)
redis_client = aioredis.Redis(connection_pool=redis_pool)

class RedisSessionMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: FastAPI, session_cookie: str = "session_redis_id", max_age: int = 3600):
        super().__init__(app)
        self.session_cookie = session_cookie
        self.max_age = max_age

    async def dispatch(self, request: Request, call_next):
        response = None 
        session_id = request.cookies.get(self.session_cookie)
        initial_session_was_empty = True
        
        if self.max_age is None or self.max_age <= 0:
            return await call_next(request)
            
        try:
            if session_id:
                session_data = await redis_client.get(session_id)
                if session_data:
                    request.state.session = json.loads(session_data)
                    await redis_client.expire(session_id, self.max_age)
                    initial_session_was_empty = False
                else:
                    request.state.session = {}
            else:
                session_id = str(uuid.uuid4())
                request.state.session = {}

            response = await call_next(request)
            if request.state.session:
                is_https = request.url.scheme == 'https'

                response.set_cookie(self.session_cookie, session_id, max_age=self.max_age, httponly=True,
                                    secure=is_https, samesite='None' if is_https else 'Lax')
                await redis_client.setex(session_id, self.max_age, json.dumps(request.state.session))
                
            else:
                if not initial_session_was_empty:
                    await redis_client.delete(session_id)
                    response.delete_cookie(self.session_cookie)
        except Exception as e:
            logging.critical("error in redis session middleware:" + str(e))
        
        return response
                
        