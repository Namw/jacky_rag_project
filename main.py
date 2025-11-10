# main.py
import redis
import uvicorn
from dotenv import load_dotenv
load_dotenv()
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.model_config import ModelProvider
from src.api import auth, documents, chat, chromadb_lib, usage_limits
from contextlib import asynccontextmanager
from src.api.chat import init_chat_service
from src.services.retrieval_service import init_semantic_cache
from src.services.usage_limiter import init_usage_limiter
from src.services.scheduler_service import init_scheduler, stop_scheduler

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    print("=" * 60)
    print("🚀 RAG 应用启动中...")
    print("=" * 60)

    # ==================== 1. Redis 连接 ====================
    redis_client = None
    try:
        redis_client = redis.Redis(
            host='localhost',
            port=6379,
            decode_responses=False
        )
        redis_client.ping()
        print("✅ Redis 连接成功")
    except redis.ConnectionError as e:
        print(f"❌ Redis 连接失败: {e}")
        print("⚠️  语义缓存功能将不可用")
    except Exception as e:
        print(f"❌ Redis 初始化异常: {e}")

    # ==================== 2. 语义缓存初始化 ====================
    if redis_client:
        try:
            init_semantic_cache(
                redis_client=redis_client,
                similarity_threshold=0.8,  # 👈 自定义阈值
                ttl_hours=5  # 👈 自定义过期时间
            )
            print("✅ 语义缓存初始化成功 (阈值=0.8, TTL=5h)")
        except Exception as e:
            print(f"⚠️  语义缓存初始化失败: {e}")
    else:
        print("⚠️  跳过语义缓存初始化（Redis 不可用）")

    # ==================== 3. 聊天服务初始化 ====================
    try:
        init_chat_service(
            model_provider=ModelProvider.DEEPSEEK,
            temperature=0.7
        )
        print("✅ 聊天服务初始化成功 (模型=DeepSeek)")
    except Exception as e:
        print(f"⚠️  聊天服务初始化失败: {e}")
        print("   将在首次请求时懒加载")

    # ==================== 4. 初始化使用限额管理器 ====================
    init_usage_limiter(redis_client)

    # ==================== 5. 初始化定时任务调度器 ====================
    init_scheduler()

    print("=" * 60)
    print("🎉 应用启动完成！")
    print("=" * 60)
    print()

    yield  # 应用运行期间

    # ==================== Shutdown ====================
    print()
    print("=" * 60)
    print("🛑 应用关闭中...")
    print("=" * 60)

    # 停止定时任务
    stop_scheduler()

    if redis_client:
        try:
            redis_client.close()
            print("✅ Redis 连接已关闭")
        except Exception as e:
            print(f"⚠️  关闭 Redis 时出错: {e}")

    print("✅ 应用已安全关闭")
    print("=" * 60)

# 创建 FastAPI 应用
app = FastAPI(
    title="RAG Project API",
    description="RAG项目",
    version="1.0.0",
    lifespan=lifespan
)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)
app.include_router(chromadb_lib.router)
app.include_router(usage_limits.router)


@app.get("/")
def read_root():
    return {"message": "Welcome to RAG API! Visit /docs for documentation."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)