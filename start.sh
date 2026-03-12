#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [[ ! -f "init_project.sh" || ! -f "start_server.sh" ]]; then
  echo "❌ 缺少 init_project.sh 或 start_server.sh"
  exit 1
fi

bash "./init_project.sh"
bash "./start_server.sh"
