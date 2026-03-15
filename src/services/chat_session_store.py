"""
会话存储服务（优先 Redis，失败时降级到内存）
"""
import json
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any

import redis


class ChatSessionStore:
    """聊天会话存储"""

    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis = redis_client
        self._memory_sessions: Dict[str, Dict[str, dict]] = {}
        self._memory_messages: Dict[str, Dict[str, List[dict]]] = {}

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()

    @staticmethod
    def _safe_decode(value: Any):
        if isinstance(value, bytes):
            return value.decode("utf-8")
        return value

    def _sessions_key(self, user_id: str) -> str:
        return f"chat:user:{user_id}:sessions"

    def _meta_key(self, user_id: str, session_id: str) -> str:
        return f"chat:session:{user_id}:{session_id}:meta"

    def _messages_key(self, user_id: str, session_id: str) -> str:
        return f"chat:session:{user_id}:{session_id}:messages"

    def create_session(
        self,
        user_id: str,
        title: Optional[str] = None,
        document_id: Optional[str] = None,
        system_prompt: Optional[str] = None
    ) -> dict:
        session_id = str(uuid.uuid4())
        now = self._now_iso()
        session = {
            "session_id": session_id,
            "title": title or "新会话",
            "document_id": document_id,
            "system_prompt": system_prompt,
            "created_at": now,
            "updated_at": now,
        }

        if self.redis:
            self.redis.sadd(self._sessions_key(user_id), session_id)
            self.redis.set(
                self._meta_key(user_id, session_id),
                json.dumps(session, ensure_ascii=False)
            )
        else:
            self._memory_sessions.setdefault(user_id, {})[session_id] = session
            self._memory_messages.setdefault(user_id, {})[session_id] = []

        return session

    def list_sessions(self, user_id: str) -> List[dict]:
        sessions: List[dict] = []

        if self.redis:
            session_ids = self.redis.smembers(self._sessions_key(user_id))
            for raw_id in session_ids:
                session_id = self._safe_decode(raw_id)
                raw_meta = self.redis.get(self._meta_key(user_id, session_id))
                if raw_meta:
                    sessions.append(json.loads(self._safe_decode(raw_meta)))
        else:
            sessions = list(self._memory_sessions.get(user_id, {}).values())

        sessions.sort(key=lambda item: item.get("updated_at", ""), reverse=True)
        return sessions

    def get_session(self, user_id: str, session_id: str) -> Optional[dict]:
        if self.redis:
            raw_meta = self.redis.get(self._meta_key(user_id, session_id))
            if not raw_meta:
                return None
            return json.loads(self._safe_decode(raw_meta))

        return self._memory_sessions.get(user_id, {}).get(session_id)

    def update_session(self, user_id: str, session_id: str, updates: dict) -> Optional[dict]:
        session = self.get_session(user_id, session_id)
        if not session:
            return None

        session.update(updates)
        session["updated_at"] = self._now_iso()

        if self.redis:
            self.redis.set(
                self._meta_key(user_id, session_id),
                json.dumps(session, ensure_ascii=False)
            )
        else:
            self._memory_sessions.setdefault(user_id, {})[session_id] = session

        return session

    def delete_session(self, user_id: str, session_id: str) -> bool:
        if not self.get_session(user_id, session_id):
            return False

        if self.redis:
            self.redis.srem(self._sessions_key(user_id), session_id)
            self.redis.delete(self._meta_key(user_id, session_id))
            self.redis.delete(self._messages_key(user_id, session_id))
        else:
            self._memory_sessions.get(user_id, {}).pop(session_id, None)
            self._memory_messages.get(user_id, {}).pop(session_id, None)

        return True

    def append_message(
        self,
        user_id: str,
        session_id: str,
        role: str,
        content: str,
        sources: Optional[List[dict]] = None
    ) -> dict:
        message = {
            "message_id": str(uuid.uuid4()),
            "role": role,
            "content": content,
            "sources": sources,
            "timestamp": self._now_iso(),
        }

        if self.redis:
            self.redis.rpush(
                self._messages_key(user_id, session_id),
                json.dumps(message, ensure_ascii=False)
            )
        else:
            self._memory_messages.setdefault(user_id, {}).setdefault(session_id, []).append(message)

        self.update_session(user_id, session_id, {})
        return message

    def list_messages(self, user_id: str, session_id: str, limit: Optional[int] = None) -> List[dict]:
        if self.redis:
            total = self.redis.llen(self._messages_key(user_id, session_id))
            if total <= 0:
                return []

            if limit is None:
                start = 0
            else:
                start = max(0, total - limit)

            raw_items = self.redis.lrange(self._messages_key(user_id, session_id), start, -1)
            return [json.loads(self._safe_decode(item)) for item in raw_items]

        all_messages = self._memory_messages.get(user_id, {}).get(session_id, [])
        if limit is None:
            return all_messages
        return all_messages[-limit:]


_chat_session_store: Optional[ChatSessionStore] = None


def init_chat_session_store(redis_client: Optional[redis.Redis]):
    """初始化会话存储"""
    global _chat_session_store
    _chat_session_store = ChatSessionStore(redis_client)


def get_chat_session_store() -> ChatSessionStore:
    """获取会话存储实例"""
    global _chat_session_store
    if _chat_session_store is None:
        _chat_session_store = ChatSessionStore(None)
    return _chat_session_store
