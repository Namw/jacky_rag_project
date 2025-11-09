# src/services/usage_limiter.py
"""
使用限额管理器 - 基于 Redis
"""
import redis
import os
from datetime import datetime, date, timedelta, time
from typing import Tuple, Dict

class UsageLimitsConfig:
    """使用限制配置"""

    # 每日限额（默认值）
    DEFAULT_UPLOAD_LIMIT = int(os.getenv("DAILY_UPLOAD_LIMIT", "10"))
    DEFAULT_QUERY_LIMIT = int(os.getenv("DAILY_QUERY_LIMIT", "10"))

    # 限额管理密码
    LIMITS_ADMIN_PASSWORD = os.getenv("LIMITS_ADMIN_PASSWORD", "change_me_in_production")

    # 每日重置时间
    RESET_HOUR = int(os.getenv("RESET_HOUR", "6"))  # 早上6点
    RESET_MINUTE = int(os.getenv("RESET_MINUTE", "0"))

    # Redis key 前缀
    REDIS_KEY_PREFIX_UPLOAD = "usage:upload"
    REDIS_KEY_PREFIX_QUERY = "usage:query"
    REDIS_KEY_CONFIG_UPLOAD = "config:limits:upload"
    REDIS_KEY_CONFIG_QUERY = "config:limits:query"

usage_limits_config = UsageLimitsConfig()

class UsageLimiter:
    """使用限额管理器"""

    def __init__(self, redis_client: redis.Redis):
        """
        初始化限额管理器

        :param redis_client: Redis 客户端
        """
        self.redis = redis_client
        self.config = usage_limits_config

        # 初始化默认配置
        self._init_config()

    def _init_config(self):
        """初始化默认配置到 Redis"""
        # 如果配置不存在，设置默认值
        if not self.redis.exists(self.config.REDIS_KEY_CONFIG_UPLOAD):
            self.redis.set(
                self.config.REDIS_KEY_CONFIG_UPLOAD,
                self.config.DEFAULT_UPLOAD_LIMIT
            )

        if not self.redis.exists(self.config.REDIS_KEY_CONFIG_QUERY):
            self.redis.set(
                self.config.REDIS_KEY_CONFIG_QUERY,
                self.config.DEFAULT_QUERY_LIMIT
            )

        print(f"✅ 使用限额配置初始化完成")
        print(f"   - 上传限额: {self.get_upload_limit()}")
        print(f"   - 问答限额: {self.get_query_limit()}")

    # ==================== 限额配置管理 ====================

    def get_upload_limit(self) -> int:
        """获取上传限额"""
        limit = self.redis.get(self.config.REDIS_KEY_CONFIG_UPLOAD)
        return int(limit) if limit else self.config.DEFAULT_UPLOAD_LIMIT

    def get_query_limit(self) -> int:
        """获取问答限额"""
        limit = self.redis.get(self.config.REDIS_KEY_CONFIG_QUERY)
        return int(limit) if limit else self.config.DEFAULT_QUERY_LIMIT

    def update_limits(self, upload_limit: int, query_limit: int):
        """
        更新限额配置

        :param upload_limit: 新的上传限额
        :param query_limit: 新的问答限额
        """
        self.redis.set(self.config.REDIS_KEY_CONFIG_UPLOAD, upload_limit)
        self.redis.set(self.config.REDIS_KEY_CONFIG_QUERY, query_limit)

        print(f"📝 限额已更新: 上传={upload_limit}, 问答={query_limit}")

    # ==================== Redis Key 生成 ====================

    def _make_upload_key(self, user_id: int) -> str:
        """生成上传计数的 Redis key"""
        today = date.today().isoformat()
        return f"{self.config.REDIS_KEY_PREFIX_UPLOAD}:{user_id}:{today}"

    def _make_query_key(self, user_id: int) -> str:
        """生成问答计数的 Redis key"""
        today = date.today().isoformat()
        return f"{self.config.REDIS_KEY_PREFIX_QUERY}:{user_id}:{today}"

    def _calculate_ttl(self) -> int:
        """计算 TTL（到明天重置时间的秒数）"""
        now = datetime.now()

        # 明天的重置时间
        tomorrow = now.date() + timedelta(days=1)
        reset_time = datetime.combine(
            tomorrow,
            time(self.config.RESET_HOUR, self.config.RESET_MINUTE)
        )

        # 计算秒数
        ttl = int((reset_time - now).total_seconds())
        return ttl

    # ==================== 上传限额管理 ====================

    def get_upload_count(self, user_id: int) -> int:
        """获取当前上传次数"""
        key = self._make_upload_key(user_id)
        count = self.redis.get(key)
        return int(count) if count else 0

    def check_can_upload(self, user_id: int) -> Tuple[bool, str]:
        """
        检查是否可以上传

        :return: (是否可以, 错误信息)
        """
        current_count = self.get_upload_count(user_id)
        limit = self.get_upload_limit()

        if current_count >= limit:
            return False, f"今日上传次数已达上限（{limit} 次），明天 {self.config.RESET_HOUR:02d}:{self.config.RESET_MINUTE:02d} 重置"

        return True, ""

    def increment_upload(self, user_id: int) -> int:
        """
        增加上传次数（原子操作）

        :return: 增加后的计数
        """
        key = self._make_upload_key(user_id)

        # Redis INCR 是原子操作
        new_count = self.redis.incr(key)

        # 如果是第一次创建，设置过期时间
        if new_count == 1:
            ttl = self._calculate_ttl()
            self.redis.expire(key, ttl)
            print(f"📊 创建上传计数器: {key}, TTL={ttl}秒")

        return new_count

    # ==================== 问答限额管理 ====================

    def get_query_count(self, user_id: int) -> int:
        """获取当前问答次数"""
        key = self._make_query_key(user_id)
        count = self.redis.get(key)
        return int(count) if count else 0

    def check_can_query(self, user_id: int) -> Tuple[bool, str]:
        """
        检查是否可以问答

        :return: (是否可以, 错误信息)
        """
        current_count = self.get_query_count(user_id)
        limit = self.get_query_limit()

        if current_count >= limit:
            return False, f"今日问答次数已达上限（{limit} 次），明天 {self.config.RESET_HOUR:02d}:{self.config.RESET_MINUTE:02d} 重置"

        return True, ""

    def increment_query(self, user_id: int) -> int:
        """
        增加问答次数（原子操作）

        :return: 增加后的计数
        """
        key = self._make_query_key(user_id)

        # Redis INCR 是原子操作
        new_count = self.redis.incr(key)

        # 如果是第一次创建，设置过期时间
        if new_count == 1:
            ttl = self._calculate_ttl()
            self.redis.expire(key, ttl)
            print(f"📊 创建问答计数器: {key}, TTL={ttl}秒")

        return new_count

    # ==================== 统计信息 ====================

    def get_usage_stats(self, user_id: int) -> Dict:
        """
        获取使用统计

        :return: 统计信息字典
        """
        upload_count = self.get_upload_count(user_id)
        query_count = self.get_query_count(user_id)

        upload_limit = self.get_upload_limit()
        query_limit = self.get_query_limit()

        # 计算下次重置时间
        now = datetime.now()
        tomorrow = now.date() + timedelta(days=1)
        next_reset = datetime.combine(
            tomorrow,
            time(self.config.RESET_HOUR, self.config.RESET_MINUTE)
        )

        return {
            "upload_count": upload_count,
            "query_count": query_count,
            "upload_limit": upload_limit,
            "query_limit": query_limit,
            "upload_remaining": max(0, upload_limit - upload_count),
            "query_remaining": max(0, query_limit - query_count),
            "next_reset_time": next_reset.isoformat()
        }

    # ==================== 调试工具 ====================

    def reset_user_usage(self, user_id: int):
        """重置用户的使用计数（调试用）"""
        upload_key = self._make_upload_key(user_id)
        query_key = self._make_query_key(user_id)

        self.redis.delete(upload_key, query_key)
        print(f"🗑️ 已重置用户 {user_id} 的使用计数")


# ==================== 全局实例 ====================
_usage_limiter = None


def init_usage_limiter(redis_client: redis.Redis):
    """初始化使用限额管理器"""
    global _usage_limiter
    _usage_limiter = UsageLimiter(redis_client)
    print("✅ 使用限额管理器初始化成功")


def get_usage_limiter() -> UsageLimiter:
    """获取使用限额管理器实例"""
    if _usage_limiter is None:
        raise RuntimeError("UsageLimiter 未初始化，请先调用 init_usage_limiter()")
    return _usage_limiter