# src/services/scheduler_service.py
"""
定时任务调度服务 - 集中管理所有定时任务
"""
import os
import shutil
from pathlib import Path
from datetime import datetime
from apscheduler.schedulers.background import BackgroundScheduler

# ==================== 全局实例 ====================
_scheduler = None


# ==================== 定时任务函数 ====================

def _reset_limits_daily():
    """定时任务：重置限额到默认值"""
    try:
        from src.services.usage_limiter import get_usage_limiter

        limiter = get_usage_limiter()
        limiter.reset_limits_to_default()
        print(f"✅ [定时任务] {datetime.now().isoformat()} - 限额已重置到默认值")

    except Exception as e:
        print(f"❌ [定时任务] 限额重置失败: {e}")


def _clear_semantic_cache_daily():
    """定时任务：清理语义查询缓存"""
    try:
        from src.services.retrieval_service import get_semantic_cache

        cache = get_semantic_cache()
        if cache is None:
            print("⚠️  [定时任务] 语义缓存未初始化，跳过清理")
            return

        # 清理所有缓存
        cache.clear_cache(collection_name=None)
        print(f"✅ [定时任务] {datetime.now().isoformat()} - 语义缓存已清理")

    except Exception as e:
        print(f"❌ [定时任务] 缓存清理失败: {e}")


def _cleanup_upload_files_daily():
    """定时任务：清理过期的上传文件 ⭐️ 新增"""
    try:
        upload_dir = Path("data/uploads")

        if not upload_dir.exists():
            print("⚠️  [定时任务] 上传文件目录不存在，跳过清理")
            return

        # 获取目录下所有文件
        files = list(upload_dir.glob("*"))

        if not files:
            print(f"✅ [定时任务] {datetime.now().isoformat()} - 上传文件目录已为空，无需清理")
            return

        # 删除所有文件
        deleted_count = 0
        total_size = 0

        for file_path in files:
            try:
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_path.unlink()  # 删除文件
                    deleted_count += 1
                    total_size += file_size
                elif file_path.is_dir():
                    shutil.rmtree(file_path)  # 删除目录
                    deleted_count += 1
            except Exception as e:
                print(f"⚠️  [定时任务] 删除文件失败 {file_path}: {e}")

        # 格式化文件大小
        size_mb = total_size / (1024 * 1024)

        print(f"✅ [定时任务] {datetime.now().isoformat()} - 上传文件已清理")
        print(f"   - 已删除: {deleted_count} 个文件/目录")
        print(f"   - 释放空间: {size_mb:.2f} MB")

    except Exception as e:
        print(f"❌ [定时任务] 上传文件清理失败: {e}")


# ==================== 调度器初始化 ====================

def init_scheduler():
    """初始化定时任务调度器 ⭐️ 新增"""
    global _scheduler

    try:
        _scheduler = BackgroundScheduler()

        # 从环境变量获取重置时间配置
        reset_hour = int(os.getenv("RESET_HOUR", "6"))
        reset_minute = int(os.getenv("RESET_MINUTE", "0"))

        # 1️⃣ 限额重置任务
        _scheduler.add_job(
            func=_reset_limits_daily,
            trigger="cron",
            hour=reset_hour,
            minute=reset_minute,
            id="reset_limits_daily",
            name="每日限额重置任务",
            timezone="Asia/Shanghai"
        )

        # 2️⃣ 缓存清理任务
        _scheduler.add_job(
            func=_clear_semantic_cache_daily,
            trigger="cron",
            hour=reset_hour,
            minute=reset_minute,
            id="clear_cache_daily",
            name="每日缓存清理任务",
            timezone="Asia/Shanghai"
        )

        # 3️⃣ 上传文件清理任务 ⭐️ 新增
        _scheduler.add_job(
            func=_cleanup_upload_files_daily,
            trigger="cron",
            hour=reset_hour,
            minute=reset_minute,
            id="cleanup_upload_files_daily",
            name="每日上传文件清理任务",
            timezone="Asia/Shanghai"
        )

        _scheduler.start()
        print(f"✅ 定时任务调度器已启动: 每天 {reset_hour:02d}:{reset_minute:02d} 执行")
        print(f"   📋 任务列表:")
        print(f"      1️⃣  限额重置")
        print(f"      2️⃣  语义缓存清理")
        print(f"      3️⃣  上传文件清理")

    except Exception as e:
        print(f"⚠️  定时任务调度器启动失败: {e}")


def stop_scheduler():
    """停止定时任务调度器（应用关闭时调用）"""
    global _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown()
        print("✅ 定时任务调度器已停止")


def get_scheduler():
    """获取调度器实例（用于测试或监控）"""
    return _scheduler


def get_scheduler_status():
    """获取调度器状态（用于调试）"""
    if not _scheduler:
        return {
            "status": "not initialized",
            "jobs": []
        }

    if not _scheduler.running:
        return {
            "status": "stopped",
            "jobs": []
        }

    jobs_info = []
    for job in _scheduler.get_jobs():
        jobs_info.append({
            "id": job.id,
            "name": job.name,
            "next_run_time": str(job.next_run_time) if job.next_run_time else None
        })

    return {
        "status": "running",
        "jobs": jobs_info
    }
