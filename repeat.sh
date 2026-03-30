#!/bin/bash

# Usage: ./repeat.sh <command> [args...]
# Example: ./repeat.sh echo "hello"

if [ $# -eq 0 ]; then
    echo "Usage: $0 <command> [args...]"
    echo "Example: $0 echo hello"
    exit 1
fi

echo "Running: $@  (every 0.5s) — Press Ctrl+C to stop"
echo "---------------------------------------------------"

while true; do
    "$@"
    sleep 0.5
done