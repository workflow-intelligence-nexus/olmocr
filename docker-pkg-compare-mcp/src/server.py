#!/usr/bin/env python3
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from modelcontextprotocol import McpServer
from pydantic import BaseModel
from typing import Optional
import subprocess
import os

app = FastAPI()
mcp = McpServer(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class CompareRequest(BaseModel):
    dockerfile_path: str
    requirements_path: Optional[str] = None
    compose_path: Optional[str] = None
    conda_env: Optional[str] = None

@mcp.tool(
    name="compare_environments",
    description="Compare package versions between Docker and local environment",
    input_model=CompareRequest
)
async def compare_environments(request: CompareRequest):
    """Compare Docker container packages with local environment"""
    from src.compare import compare_packages
    
    result = compare_packages(
        request.dockerfile_path,
        request.requirements_path,
        request.compose_path,
        request.conda_env
    )
    return {"status": "success", "data": result}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
