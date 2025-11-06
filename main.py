# main.py
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from config.model_config import ModelProvider
from src.api import auth, documents, chat
from contextlib import asynccontextmanager
from src.api.chat import init_chat_service

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 可选：预热 LLM
    try:
        init_chat_service(
            model_provider=ModelProvider.DEEPSEEK,
            temperature=0.7
        )
        print("✅ Chat service pre-initialized")
    except Exception as e:
        print(f"⚠️  Failed to pre-initialize: {e}")
        print("   Will use lazy initialization instead")

    yield


# 创建 FastAPI 应用
app = FastAPI(
    title="RAG Project API",
    description="用于RAG面试项目的后端API",
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


@app.get("/")
def read_root():
    return {"message": "Welcome to RAG API! Visit /docs for documentation."}


if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)