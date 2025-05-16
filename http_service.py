from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import tempfile
import shutil
import asyncio
from typing import Optional
import logging
import subprocess
import time
import httpx
import os

# Import processing functions from pipeline
from olmocr.pipeline import process_pdf_file_async, SGLANG_SERVER_PORT

# Global variable to track server process
sglang_server = None

app = FastAPI(
    title="PDF Processing Service",
    description="API for processing PDF files through OLMoCR pipeline",
    version="0.1.0"
)

logger = logging.getLogger(__name__)

@app.on_event("startup")
async def startup_event():
    """Check if the external SGLang server is available"""
    try:
        # Get the SGLang server URL from environment or use default
        sglang_url = os.environ.get("SGLANG_SERVER_URL", f"http://sglang:30024")
        logger.info(f"Using SGLang server at: {sglang_url}")
        
        # Check if the server is ready
        timeout = 120  # Increased timeout for GPU initialization
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with httpx.Client() as client:
                    response = client.get(f"{sglang_url}/v1/models")
                    if response.status_code == 200:
                        logger.info("SGLang server is available and ready")
                        return
            except Exception as conn_err:
                logger.debug(f"Waiting for SGLang server: {str(conn_err)}")
                time.sleep(2)
                
        logger.warning("SGLang server not responding within timeout, but continuing startup")
    except Exception as e:
        logger.error(f"Error checking SGLang server: {str(e)}")
        # We don't want to fail startup completely

@app.post("/process-pdf")
async def process_pdf(file: UploadFile):
    """Process uploaded PDF file"""
    try:
        if file.content_type != "application/pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
            shutil.copyfileobj(file.file, temp_pdf)
            temp_path = temp_pdf.name

        # Process the PDF using the async version
        result = await process_pdf_file_async(temp_path)
        
        # Clean up temp file
        Path(temp_path).unlink(missing_ok=True)

        return JSONResponse(content={
            "status": "success",
            "filename": file.filename,
            "result": result
        })

    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Service health check endpoint"""
    return {"status": "healthy"}
