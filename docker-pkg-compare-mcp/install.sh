#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Make scripts executable
chmod +x src/server.py

echo "Docker Package Compare MCP Server installed successfully"
echo "To start the server: python src/server.py"
