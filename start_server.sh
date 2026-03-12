#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE=".env"
LOG_DIR="log"
LOG_FILE="$LOG_DIR/app.log"
PORT=8000

if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ 未找到 $ENV_FILE，请先运行 ./init_project.sh"
  exit 1
fi

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

EXISTING_PID=""
if command -v lsof >/dev/null 2>&1; then
  EXISTING_PID="$(lsof -tiTCP:${PORT} -sTCP:LISTEN 2>/dev/null | head -n 1 || true)"
elif command -v ss >/dev/null 2>&1; then
  EXISTING_PID="$(ss -lntp 2>/dev/null | awk -v port=":${PORT}" '$4 ~ port { if (match($0, /pid=[0-9]+/)) { print substr($0, RSTART + 4, RLENGTH - 4); exit } }' || true)"
fi

if [[ -n "$EXISTING_PID" ]]; then
  echo "⚠️  检测到端口 ${PORT} 被占用，正在停止进程 PID=${EXISTING_PID}"
  kill -15 "$EXISTING_PID" 2>/dev/null || true
  sleep 1

  if kill -0 "$EXISTING_PID" 2>/dev/null; then
    echo "⚠️  进程仍在运行，强制停止 PID=${EXISTING_PID}"
    kill -9 "$EXISTING_PID"
    sleep 1
  fi
fi

nohup uv run uvicorn main:app --host 0.0.0.0 --port "$PORT" > "$LOG_FILE" 2>&1 &
PID=$!

echo "🚀 服务已启动，PID=$PID"
echo "📄 日志查看：tail -f $LOG_FILE"
