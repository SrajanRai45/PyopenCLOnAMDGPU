from contextlib import asynccontextmanager
from fastapi import FastAPI
from src.api.calculate import arrayCalclate
from src.services.pyopenclcompute import OpenCLCalculatorService

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.gpu_service = OpenCLCalculatorService()
    yield
    # Any necessary cleanup code can go here during shutdown
    del app.state.gpu_service

app = FastAPI(lifespan=lifespan)

app.include_router(arrayCalclate)
