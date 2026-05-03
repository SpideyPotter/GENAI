#!/usr/bin/env bash
# Serve AgriChat on 127.0.0.1. Tries several ports if the default is busy (shared hosts).
set -u
DIR="$(cd "$(dirname "$0")" && pwd)"

port_free() {
  python3 -c "import socket; s=socket.socket(); s.bind(('127.0.0.1', int('$1'))); s.close()" 2>/dev/null
}

if [ -n "${1:-}" ]; then
  echo "Open: http://127.0.0.1:$1/"
  exec python3 -m http.server "$1" --bind 127.0.0.1 --directory "$DIR"
fi

for p in 8765 8766 8767 8778 8888 9898 18080 29333 40123 51234; do
  if port_free "$p"; then
    echo "Open: http://127.0.0.1:$p/"
    exec python3 -m http.server "$p" --bind 127.0.0.1 --directory "$DIR"
  fi
done

echo "No free port in the built-in list. Pick one and run:" >&2
echo "  $0 9123" >&2
exit 1
