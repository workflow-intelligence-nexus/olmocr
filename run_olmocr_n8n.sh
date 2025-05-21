#!/bin/sh
# Simple wrapper for n8n

# Log start
echo "$(date) - Starting olmocr server" > /tmp/olmocr_n8n.log

# Change to script directory
cd /data/shared/olmocrLinux || {
    echo "Failed to change directory" >> /tmp/olmocr_n8n.log
    exit 1
}

# Run the actual script with logging
sh ./startOlmocrInBkgd.sh >> /tmp/olmocr_n8n.log 2>&1 &
echo $! > /tmp/olmocr_n8n.pid

# Verify it's running
if kill -0 $(cat /tmp/olmocr_n8n.pid) 2>/dev/null; then
    echo "Server started successfully" >> /tmp/olmocr_n8n.log
    echo "Server started with PID $(cat /tmp/olmocr_n8n.pid)"
    exit 0
else
    echo "Failed to start server" >> /tmp/olmocr_n8n.log
    exit 1
fi
