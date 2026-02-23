from fastapi import FastAPI
from src.api.calculate import arrayCalclate

app = FastAPI()

app.include_router(arrayCalclate)