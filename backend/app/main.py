from fastapi import FastAPI
from app.pipeline import Pipeline

app = FastAPI()
pipeline = Pipeline()

@app.post("/analyze")
async def analyze(input_type: str, data: str):
    return pipeline.run(input_type, data)

@app.get("/")
async def health():    
    return {"status": "ok"}
