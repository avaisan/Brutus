#!/bin/bash
# Start the Raspberry Pi terminal client in the foreground. Run from the repository root.

PIDDIR="/tmp/brutus"
PI_CLIENT_PID="$PIDDIR/pi_client.pid"
mkdir -p "$PIDDIR"

if [ -f "$PI_CLIENT_PID" ] && kill -0 "$(cat "$PI_CLIENT_PID")" 2>/dev/null; then
    echo "Raspberry Pi client is already running. Run scripts/stop.sh first."
    exit 1
fi

cd raspberrypi_client || exit 1
echo $$ > "$PI_CLIENT_PID"
trap 'rm -f "$PI_CLIENT_PID"' EXIT
uv run task start
