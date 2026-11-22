#!/bin/bash
# Start the backend (port 8000) and, once it is healthy, the frontend (port 3000).
# Run from the repository root. Stop with Ctrl-C or scripts/stop.sh.

PIDDIR="/tmp/brutus"
BACKEND_PID="$PIDDIR/backend.pid"
FRONTEND_PID="$PIDDIR/frontend.pid"
mkdir -p "$PIDDIR"

is_port_in_use() {
    ss -ltn "sport = :$1" | grep -q LISTEN
}

stop_service() {
    if [ -f "$1" ]; then
        kill "$(cat "$1")" 2>/dev/null
        rm -f "$1"
    fi
}

cleanup() {
    echo "Stopping services..."
    stop_service "$FRONTEND_PID"
    stop_service "$BACKEND_PID"
    exit 0
}

wait_for_backend() {
    echo "Waiting for backend to be healthy..."
    for attempt in $(seq 1 15); do
        if curl -s http://localhost:8000/health | grep -q '"status":"Healthy"'; then
            echo "Backend is healthy!"
            return 0
        fi
        echo "Attempt $attempt/15: backend not ready yet..."
        sleep 3
    done
    return 1
}

trap cleanup SIGINT SIGTERM

for port in 8000 3000; do
    if is_port_in_use "$port"; then
        echo "Port $port is already in use. Run scripts/stop.sh first."
        exit 1
    fi
done

echo "Starting backend..."
(cd backend && exec uv run task start) &
echo $! > "$BACKEND_PID"

if ! wait_for_backend; then
    echo "Backend failed to become healthy"
    cleanup
fi

echo "Starting frontend..."
(cd frontend && exec uv run task start) &
echo $! > "$FRONTEND_PID"

echo "Services started: http://$(hostname -I | awk '{print $1}'):3000"
wait
