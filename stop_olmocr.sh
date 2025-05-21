#!/bin/bash
if [ -f olmocr_server.pid ]; then
    echo "Stopping olmocr server (PID: $(cat olmocr_server.pid))..."
    kill -TERM $(cat olmocr_server.pid) 2>/dev/null
    rm -f olmocr_server.pid
    echo "Server stopped."
else
    echo "No running olmocr server found (no PID file)."
fi