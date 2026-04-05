from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from typing import Optional
from app.pipeline import Pipeline
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()
pipeline = Pipeline()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(exc.body)},
    )

@app.post("/analyze")
async def analyze(
    type: str = Form("image"),
    file: Optional[UploadFile] = File(None),
    url: Optional[str] = Form(None)
):
    logger.info(f"Received analysis request: type={type}, file={file.filename if file else 'None'}, url={url}")
    return await pipeline.run(type, file=file, url=url)

@app.get("/")
async def health():    
    return {"status": "ok"}
