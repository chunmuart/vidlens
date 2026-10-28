"""锁续期（心跳）。

背景：任务锁原本是 `SET NX EX 7200`——固定两小时。
如果视频很长（转写 + OCR + Agent 超过两小时），锁会先过期，
第二个请求就能拿到锁，出现两个执行者。

这里不引入后台线程，而是「**在进度回调里顺手续期**」：
只要任务还在推进（转写、OCR、Agent 各阶段都会回写进度），
锁的 TTL 就会被重新拉长；任务一旦卡死，不再有进度，锁自然过期。
这比定时器简单，也不增加线程和生命周期管理。

用法::

    heartbeat = LockHeartbeat(client, key, task_id, ttl_seconds=7200)
    ...
    heartbeat.touch()     # 进度回调里调用，内部有节流
"""

from __future__ import annotations

import time
from typing import Any, Callable


class LockHeartbeat:
    """按需续期，带节流，失败不抛异常。"""

    def __init__(
        self,
        client: Any,
        key: str,
        token: str,
        *,
        ttl_seconds: int = 7200,
        min_interval_seconds: float = 5.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.client = client
        self.key = key
        self.token = token
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self._clock = clock or time.monotonic
        self._last_touch: float | None = None
        self.touch_count = 0
        self.last_error: str | None = None

    def touch(self, *, force: bool = False) -> bool:
        """续期一次。返回是否真的发出了 expire。

        - 客户端为空（未配 Redis）→ 直接 False，不报错
        - 距上次续期不足 min_interval 且非 force → 跳过（节流）
        - 锁已不是自己持有 → 不再续期（避免给别人的锁续命）
        """
        if self.client is None:
            return False
        now = self._clock()
        if (
            not force
            and self._last_touch is not None
            and (now - self._last_touch) < self.min_interval_seconds
        ):
            return False
        try:
            current = self.client.get(self.key)
            if isinstance(current, bytes):
                current = current.decode("utf-8", "ignore")
            if current != self.token:
                return False
            self.client.expire(self.key, self.ttl_seconds)
            self._last_touch = now
            self.touch_count += 1
            return True
        except Exception as exc:  # noqa: BLE001 —— 续期失败不能影响主任务
            self.last_error = f"{exc.__class__.__name__}: {exc}"
            return False
