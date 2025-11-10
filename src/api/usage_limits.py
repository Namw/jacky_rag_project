# src/api/usage_limits.py
"""
使用限额管理 API
"""
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field
from typing import Optional

from src.api.auth import get_current_user, User
from src.services.usage_limiter import get_usage_limiter, UsageLimitsConfig

router = APIRouter(
    prefix="/api/usage",
    tags=["UsageLimits"]
)

# ==================== Pydantic 模型 ====================

class UsageStatsResponse(BaseModel):
    """使用统计响应"""
    upload_count: int = Field(..., description="今日已上传次数")
    query_count: int = Field(..., description="今日已问答次数")
    upload_limit: int = Field(..., description="上传限额")
    query_limit: int = Field(..., description="问答限额")
    upload_remaining: int = Field(..., description="上传剩余次数")
    query_remaining: int = Field(..., description="问答剩余次数")
    next_reset_time: str = Field(..., description="下次重置时间")

    class Config:
        json_schema_extra = {
            "example": {
                "upload_count": 3,
                "query_count": 7,
                "upload_limit": 10,
                "query_limit": 10,
                "upload_remaining": 7,
                "query_remaining": 3,
                "next_reset_time": "2025-11-10T06:00:00"
            }
        }


class LimitsConfigResponse(BaseModel):
    """限额配置响应"""
    upload_limit: int = Field(..., description="上传限额")
    query_limit: int = Field(..., description="问答限额")


class UpdateLimitsRequest(BaseModel):
    """更新限额请求"""
    admin_password: str = Field(..., description="管理密码")
    upload_limit: int = Field(..., ge=1, le=10000, description="新的上传限额")
    query_limit: int = Field(..., ge=1, le=10000, description="新的问答限额")

    class Config:
        json_schema_extra = {
            "example": {
                "admin_password": "YourSecretPassword123!",
                "upload_limit": 50,
                "query_limit": 100
            }
        }


class ResetUsageRequest(BaseModel):
    """重置使用记录请求"""
    admin_password: str = Field(..., description="管理密码")
    user_id: Optional[int] = Field(None, description="要重置的用户ID，不填则重置自己")

    class Config:
        json_schema_extra = {
            "example": {
                "admin_password": "YourSecretPassword123!",
                "user_id": 123
            }
        }


# ==================== API 路由 ====================

@router.get("/stats", response_model=UsageStatsResponse)
async def get_usage_stats(
        current_user: User = Depends(get_current_user)
):
    """
    获取当前用户的使用统计

    返回今日已使用次数、剩余次数、限额配置等信息
    """
    try:
        limiter = get_usage_limiter()
        stats = limiter.get_usage_stats(current_user.id)

        return UsageStatsResponse(**stats)

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取统计信息失败: {str(e)}"
        )


@router.get("/limits", response_model=LimitsConfigResponse)
async def get_limits_config(
        current_user: User = Depends(get_current_user)
):
    """
    获取当前限额配置

    返回全局的上传和问答限额设置
    """
    try:
        limiter = get_usage_limiter()

        return LimitsConfigResponse(
            upload_limit=limiter.get_upload_limit(),
            query_limit=limiter.get_query_limit()
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取限额配置失败: {str(e)}"
        )


@router.post("/limits/update")
async def update_limits(
        request: UpdateLimitsRequest,
        current_user: User = Depends(get_current_user)
):
    """
    更新全局限额配置（需要管理员 + 特殊密码）

    **权限要求：**
    - 必须是 admin 角色
    - 必须提供正确的管理密码

    **注意：** 修改后立即生效，但不影响今日已使用的计数
    """
    # 1. 检查是否是 admin
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有管理员可以修改限额配置"
        )

    # 2. 验证管理密码
    if request.admin_password != UsageLimitsConfig.LIMITS_ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="管理密码错误"
        )

    # 3. 更新限额
    try:
        limiter = get_usage_limiter()
        limiter.update_limits(
            upload_limit=request.upload_limit,
            query_limit=request.query_limit
        )

        return {
            "message": "限额配置更新成功",
            "upload_limit": request.upload_limit,
            "query_limit": request.query_limit
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"更新限额失败: {str(e)}"
        )


@router.post("/reset")
async def reset_usage(
        request: ResetUsageRequest,
        current_user: User = Depends(get_current_user)
):
    """
    重置用户的使用计数和限额配置（需要管理员 + 特殊密码）

    **权限要求：**
    - 必须是 admin 角色
    - 必须提供正确的管理密码

    **功能：**
    - 重置用户的今日计数（上传和问答都变为 0）
    - 恢复限额配置到 .env 中的默认值
    - 如果提供 user_id：重置指定用户
    - 如果不提供 user_id：重置自己

    **重置内容：** ⭐️ 新增
    - 清除用户的上传计数
    - 清除用户的问答计数
    - 恢复全局限额到默认值（DAILY_UPLOAD_LIMIT、DAILY_QUERY_LIMIT）
    """
    # 1. 检查是否是 admin
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="只有管理员可以重置使用计数"
        )

    # 2. 验证管理密码
    if request.admin_password != UsageLimitsConfig.LIMITS_ADMIN_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="管理密码错误"
        )

    # 3. 确定要重置的用户
    target_user_id = request.user_id if request.user_id else current_user.id

    # 4. 重置计数和限额
    try:
        limiter = get_usage_limiter()

        # 重置指定用户的计数
        limiter.reset_user_usage(target_user_id)

        # 恢复全局限额到默认值 ⭐️ 新增
        limiter.reset_limits_to_default()

        return {
            "message": f"已重置用户 {target_user_id} 的使用计数并恢复全局限额到默认值",
            "user_id": target_user_id,
            "reset_limits": {
                "upload_limit": limiter.get_upload_limit(),
                "query_limit": limiter.get_query_limit()
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"重置失败: {str(e)}"
        )