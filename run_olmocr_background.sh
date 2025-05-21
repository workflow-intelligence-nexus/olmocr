#!/bin/bash
# run_olmocr_background.sh - Wrapper to run olmocr in background with health checks

# Configuration
LOG_FILE="olmocr_server.log"
PID_FILE="olmocr_server.pid"
HTTP_PORT=8000  # Should match the port in startOlmocrInBkgd.sh
MAX_WAIT=300    # Max 5 minutes for server to start
CHECK_INTERVAL=2 # Check every 2 seconds

# Function to check if server is running
check_server() {
    curl -s -f "http://localhost:$HTTP_PORT/system-health" >/dev/null 2>&1
    return $?
}

# Start the server in background
echo "Starting olmocr server in background..."
nohup ./startOlmocrInBkgd.sh > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Save PID to file
echo $SERVER_PID > "$PID_FILE"
echo "Server started with PID: $SERVER_PID"
echo "Logs are being written to: $(pwd)/$LOG_FILE"

# Wait for server to be ready
echo "Waiting for server to be ready (max $MAX_WAIT seconds)..."
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if check_server; then
        echo -e "\nServer is up and running!"
        echo "Access the API at: http://localhost:$HTTP_PORT"
        exit 0
    fi
    
    # Check if server process is still running
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo -e "\nError: Server process (PID $SERVER_PID) has terminated unexpectedly."
        echo "Check the log file for errors: $LOG_FILE"
        exit 1
    fi
    
    # Print progress
    echo -n "."
    sleep $CHECK_INTERVAL
    ELAPSED=$((ELAPSED + CHECK_INTERVAL))
done

# If we get here, server didn't start in time
echo -e "\nError: Server did not become ready within $MAX_WAIT seconds."
echo "Check the log file for errors: $LOG_FILE"
exit 1