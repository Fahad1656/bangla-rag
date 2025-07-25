# main.py

from fastapi import FastAPI
from api_routes import router as api_router

app = FastAPI(title="Bangla/English RAG API")


app.include_router(api_router, prefix="/api")

@app.get("/")
def read_root():
    return {"message": "Welcome to the RAG API"}
