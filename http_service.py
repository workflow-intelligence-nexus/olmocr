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
    """Start the SGLang server when the application starts"""
    try:
        start_sglang_server()
        logger.info("SGLang server started successfully")
    except Exception as e:
        logger.error(f"Failed to start SGLang server: {str(e)}")
        # We don't want to fail startup completely, as the server might be started manually

def start_sglang_server():
    """Start SGLang server process specifically targeting the A6000 GPU"""
    global sglang_server
    try:
        # Check if server is already running
        if sglang_server is not None and sglang_server.poll() is None:
            return True
            
        # Set environment variables to target the A6000 GPU (using same settings as processPDF.sh)
        env = os.environ.copy()
        env["CUDA_HOME"] = "/usr/local/cuda-12.8"
        env["PATH"] = f"{env['CUDA_HOME']}/bin:{env.get('PATH', '')}"
        env["LD_LIBRARY_PATH"] = f"{env['CUDA_HOME']}/lib64:{env.get('LD_LIBRARY_PATH', '')}"
        env["CPLUS_INCLUDE_PATH"] = f"{env['CUDA_HOME']}/targets/x86_64-linux/include:{env.get('CPLUS_INCLUDE_PATH', '')}"
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = "1"  # Using GPU 1 as specified in processPDF.sh
        
        # Start the server with specific GPU settings
        cmd = [
            "python3", 
            "-m", 
            "sglang.launch_server",
            "--model-path", 
            "allenai/olmOCR-7B-0225-preview",
            "--port", 
            str(SGLANG_SERVER_PORT),
            "--device", 
            "cuda",
            "--dtype", 
            "half"  # Use half precision for better performance
        ]
        
        logger.info(f"Starting SGLang server with command: {' '.join(cmd)}")
        
        sglang_server = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env
        )
        
        # Wait for server to be ready
        timeout = 120  # Increased timeout for GPU initialization
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                with httpx.Client() as client:
                    response = client.get(f"http://localhost:{SGLANG_SERVER_PORT}/v1/models")
                    if response.status_code == 200:
                        logger.info("SGLang server started successfully")
                        return True
            except Exception as conn_err:
                logger.debug(f"Waiting for SGLang server: {str(conn_err)}")
                time.sleep(2)
                
        # If we get here, check if the process is still running but not responding
        if sglang_server.poll() is None:
            # Process is still running, dump stderr to help diagnose
            stderr_output = sglang_server.stderr.read().decode('utf-8', errors='replace')
            logger.error(f"SGLang server started but not responding. Error output: {stderr_output}")
        else:
            # Process exited
            return_code = sglang_server.poll()
            stderr_output = sglang_server.stderr.read().decode('utf-8', errors='replace')
            logger.error(f"SGLang server exited with code {return_code}. Error output: {stderr_output}")
                
        raise RuntimeError("SGLang server failed to start within timeout")
        
    except Exception as e:
        logger.error(f"Failed to start SGLang server: {str(e)}")
        if sglang_server:
            sglang_server.terminate()
        raise

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
