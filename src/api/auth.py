from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta, timezone
from typing import Optional

# --- 1. 配置和工具 ---

# 延迟初始化 CryptContext
_pwd_context = None


def get_pwd_context():
    """延迟初始化密码上下文,避免导入时的bcrypt问题"""
    global _pwd_context
    if _pwd_context is None:
        _pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    return _pwd_context


# 初始化 fake_user_db 的函数
def _init_fake_user_db():
    pwd_context = get_pwd_context()
    return {
        "admin": {
            "username": "admin",
            "hashed_password": pwd_context.hash("123456"),
            "role": "admin"
        }
    }


# 模拟一个数据库用户
fake_user_db = None

# JWT 配置
SECRET_KEY = "jacky_mcp_rag_20251101"  # ⭐️ 必须换成你自己的私钥!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# OAuth2 方案
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token")


# --- 2. Pydantic 模型 ---

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: str | None = None


class User(BaseModel):
    username: str
    role: str
    id: Optional[str] = None  # 添加这个字段


# --- 3. 辅助函数 ---

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    """创建 JWT Token"""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


# --- 4. ⭐️ 核心：依赖项 (用于校验 Token) ---

async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    """
    依赖项：验证Token并返回用户信息
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        # 解码 Token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    # 从数据库获取用户
    global fake_user_db
    if fake_user_db is None:
        fake_user_db = _init_fake_user_db()
    user = fake_user_db.get(token_data.username)
    if user is None:
        raise credentials_exception

    return User(
        username=user["username"],
        role=user["role"],
        id=user["username"]  # 使用 username 作为 id
    )


# --- 5. 路由 ---

router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"],
)


@router.post("/token", response_model=Token)
async def login_for_access_token(
        form_data: OAuth2PasswordRequestForm = Depends()
):
    """
    登录接口：验证用户名密码并返回Token
    """
    # 初始化用户数据库
    global fake_user_db
    if fake_user_db is None:
        fake_user_db = _init_fake_user_db()

    # 验证用户
    user = fake_user_db.get(form_data.username)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 验证密码
    pwd_context = get_pwd_context()
    if not pwd_context.verify(form_data.password, user["hashed_password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 创建 Token
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )

    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/logout")
async def logout():
    """
    登出接口：前端删除Token即可
    """
    return {"message": "Logged out successfully (frontend should clear token)"}


@router.get("/user", response_model=User)
async def get_user_info(
        current_user: User = Depends(get_current_user)
):
    """
    获取当前登录用户信息
    需要有效的Token才能访问
    """
    return current_user