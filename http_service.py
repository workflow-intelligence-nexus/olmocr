# http_service.py
import os
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from typing import Dict, Any
from olmocr.pipeline import process_pdf_file_async, metrics as pipeline_metrics # Import async version

app = FastAPI(title="OLMOCR Pipeline API",
            description="REST wrapper for olmocr PDF processing pipeline",
            version="1.0.0")

@app.post("/process-pdf", response_model=Dict[str, Any])
async def process_pdf_endpoint(pdf: UploadFile = File(..., description="PDF file to process")):
    """Process a PDF using the olmocr pipeline"""
    try:
        # Validate input
        if not pdf.filename.lower().endswith(('.pdf', '.png', '.jpg', '.jpeg')):
            raise HTTPException(status_code=400, detail="Invalid file type - must be PDF or image")

        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(pdf.filename)[1]) as tmp:
            content = await pdf.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Use existing pipeline functionality
        # Use existing pipeline functionality - directly await the async version
        result = await process_pdf_file_async(tmp_path)
        
        # Cleanup temp file
        os.unlink(tmp_path)

        if not result:
            # Log this specific condition for server-side clarity
            # Consider adding a logger instance if not already present globally
            print(f"INFO: No processable content found or document filtered out for {pdf.filename} (temp path: {tmp_path})")
            raise HTTPException(status_code=422, detail="No processable text content found in the PDF, or the document was filtered out by processing rules (e.g., too many failed pages).")
            
        return {
            "status": "success",
            "data": {
                "document_id": result["id"],
                "text": result["text"],
                "pages": result["metadata"]["pdf-total-pages"],
                "tokens": {
                    "input": result["metadata"]["total-input-tokens"],
                    "output": result["metadata"]["total-output-tokens"]
                }
            }
        }
        
    except HTTPException as http_exc:
        raise http_exc # Re-raise HTTPException
    except Exception as e:
        # It's good practice to log the exception here
        # logger.error(f"Error processing PDF: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"An unexpected error occurred: {str(e)}"}
        )

@app.get("/queue-status")
async def get_queue_status():
    """Get processing queue metrics (conceptual)"""
    # This is a placeholder. Actual queue/metrics access would depend on
    # how pipeline.py exposes this information or if a separate metrics
    # system is in place that this API can query.
    # For now, using the imported 'pipeline_metrics' if it has relevant methods.
    try:
        # These are conceptual calls. The actual methods on pipeline_metrics might differ.
        # queue_size = pipeline_metrics.get_queue_size() # Example
        # throughput = pipeline_metrics.get_tokens_per_minute() # Example
        # active_workers = pipeline_metrics.get_active_workers() # Example
        # For demonstration, returning placeholder values:
        return {
            "message": "Queue status endpoint is conceptual. Actual implementation depends on pipeline.py's metrics exposure.",
            "conceptual_metrics": {
                 "queue_size": "N/A", # Replace with actual call if available
                 "throughput_tokens_per_min": "N/A", # Replace with actual call
                 "active_workers": "N/A" # Replace with actual call
            }
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": f"Error fetching queue status: {str(e)}"}
        )


@app.get("/system-health")
async def system_health():
    """System health check endpoint, primarily for SGLang server"""
    sglang_port = os.getenv('SGLANG_SERVER_PORT', 30024)
    sglang_url = f"http://localhost:{sglang_port}/v1/models"
    try:
        import httpx
        # Using an async client within an async function
        async with httpx.AsyncClient() as client:
            response = await client.get(sglang_url)
        response.raise_for_status() # Will raise an exception for 4xx/5xx responses
        return {"status": "healthy", "sglang_server": "responsive"}
    except ImportError:
         return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "sglang_server": "httpx not installed", "error": "httpx library is required for health check."}
        )
    except httpx.RequestError as e:
        return JSONResponse(
            status_code=503, # Service Unavailable
            content={"status": "unhealthy", "sglang_server": "unreachable", "error": f"Could not connect to SGLang server at {sglang_url}: {str(e)}"}
        )
    except httpx.HTTPStatusError as e:
        return JSONResponse(
            status_code=503, # Service Unavailable
            content={"status": "unhealthy", "sglang_server": "error_response", "error": f"SGLang server at {sglang_url} responded with status {e.response.status_code}: {e.response.text}"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"status": "unhealthy", "sglang_server": "unknown_error", "error": str(e)}
        )

# To run this app (example, typically done by Uvicorn CLI):
# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
