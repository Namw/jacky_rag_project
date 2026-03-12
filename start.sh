#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

ENV_TEMPLATE=".env_copy"
ENV_FILE=".env"
LOG_DIR="log"
LOG_FILE="$LOG_DIR/app.log"

if [[ ! -f "$ENV_TEMPLATE" ]]; then
  echo "❌ 未找到 $ENV_TEMPLATE，请先确认模板文件存在。"
  exit 1
fi

# 1) 判断 .env 是否存在，不存在则创建
if [[ ! -f "$ENV_FILE" ]]; then
  cp "$ENV_TEMPLATE" "$ENV_FILE"
  echo "✅ 已创建 $ENV_FILE（来源：$ENV_TEMPLATE）"
fi

# 2) 按要求再次执行 cp .env_copy .env
cp "$ENV_TEMPLATE" "$ENV_FILE"
echo "✅ 已重置 $ENV_FILE（来源：$ENV_TEMPLATE）"

echo
echo "请依次输入以下 4 个配置值："
read -r -s -p "LIMITS_ADMIN_PASSWORD=" LIMITS_ADMIN_PASSWORD
printf "\n"
read -r -p "DASHSCOPE_API_KEY=" DASHSCOPE_API_KEY
read -r -p "OPENAI_API_KEY=" OPENAI_API_KEY
read -r -p "DEEPSEEK_API_KEY=" DEEPSEEK_API_KEY

if [[ -z "$LIMITS_ADMIN_PASSWORD" || -z "$DASHSCOPE_API_KEY" || -z "$OPENAI_API_KEY" || -z "$DEEPSEEK_API_KEY" ]]; then
  echo "❌ 4 个配置都不能为空。"
  exit 1
fi

escape_sed_replacement() {
  printf '%s' "$1" | sed -e 's/[\\/&]/\\\\&/g'
}

set_env_var() {
  local key="$1"
  local value="$2"
  local escaped_value
  escaped_value="$(escape_sed_replacement "$value")"

  if grep -qE "^${key}=" "$ENV_FILE"; then
    if [[ "$(uname -s)" == "Darwin" ]]; then
      sed -i '' "s|^${key}=.*|${key}=${escaped_value}|" "$ENV_FILE"
    else
      sed -i "s|^${key}=.*|${key}=${escaped_value}|" "$ENV_FILE"
    fi
  else
    printf '\n%s=%s\n' "$key" "$value" >> "$ENV_FILE"
  fi
}

set_env_var "LIMITS_ADMIN_PASSWORD" "$LIMITS_ADMIN_PASSWORD"
set_env_var "DASHSCOPE_API_KEY" "$DASHSCOPE_API_KEY"
set_env_var "OPENAI_API_KEY" "$OPENAI_API_KEY"
set_env_var "DEEPSEEK_API_KEY" "$DEEPSEEK_API_KEY"

echo "✅ 已写入 4 个关键配置到 $ENV_FILE"

# 3) 检查并创建 log 目录和 app.log
mkdir -p "$LOG_DIR"
touch "$LOG_FILE"
echo "✅ 日志文件就绪：$LOG_FILE"

# 4) 启动服务
nohup uv run uvicorn main:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
PID=$!

echo "🚀 服务已启动，PID=$PID"
echo "📄 日志查看：tail -f $LOG_FILE"
