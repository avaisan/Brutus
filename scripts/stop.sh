#!/bin/bash
# Stop the backend, frontend and Raspberry Pi client.

PIDDIR="/tmp/brutus"

stop() {
    local name=$1 pattern=$2 pidfile="$PIDDIR/$3"
    if [ -f "$pidfile" ]; then
        kill "$(cat "$pidfile")" 2>/dev/null
        rm -f "$pidfile"
    fi
    if pkill -f "$pattern"; then
        echo "$name stopped."
    else
        echo "$name not running."
    fi
}

echo "Stopping Brutus services..."
stop "Frontend" "uvicorn brutus_frontend.app" frontend.pid
stop "Backend" "uvicorn brutus.api" backend.pid
stop "Raspberry Pi client" "python brutus_pi.py" pi_client.pid
echo "Done."
