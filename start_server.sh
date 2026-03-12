#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_FILE=".env"
LOG_DIR="log"
LOG_FILE="$LOG_DIR/app.log"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "❌ 未找到 $ENV_FILE，请先运行 ./init_project.sh"
  exit 1
fi

mkdir -p "$LOG_DIR"
touch "$LOG_FILE"

nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
PID=$!

echo "🚀 服务已启动，PID=$PID"
echo "📄 日志查看：tail -f $LOG_FILE"
