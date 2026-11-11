from fastapi import FastAPI
from db.database import engine
from contextlib import asynccontextmanager

# Uvicorn calls the asynccontextmanager when the app starts/stops
@asynccontextmanager
async def lifespan(app: FastAPI):
    
    # ── Startup ──────────────────────────────────────────
    # Runs on the event loop; initializes the DB connection pool
    print("Starting up...")
    
    # After the yield, the app starts handling requests
    yield
    
    # ── Shutdown ─────────────────────────────────────────
    # New requests are blocked, and we get here once in-flight requests finish
    print("Shutting down...")
    
    # Release and close the whole connection pool of the SQLAlchemy async engine
    await engine.dispose()