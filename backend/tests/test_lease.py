# -*- coding: utf-8 -*-
"""锁续期（心跳）单测。"""

import unittest

from vidlens.lease import LockHeartbeat


class FakeRedis:
    def __init__(self, values=None) -> None:
        self.values = dict(values or {})
        self.expired = []

    def get(self, key):
        return self.values.get(key)

    def expire(self, key, ttl):
        self.expired.append((key, ttl))
        return True


class BrokenRedis:
    def get(self, key):
        raise RuntimeError("redis down")

    def expire(self, key, ttl):  # pragma: no cover
        return True


class FakeClock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class LockHeartbeatTest(unittest.TestCase):
    def test_touch_extends_when_token_matches(self) -> None:
        redis = FakeRedis({"lock": "task-1"})
        heartbeat = LockHeartbeat(redis, "lock", "task-1", ttl_seconds=7200, clock=FakeClock())
        self.assertTrue(heartbeat.touch())
        self.assertEqual(redis.expired, [("lock", 7200)])
        self.assertEqual(heartbeat.touch_count, 1)

    def test_touch_skips_when_token_is_not_mine(self) -> None:
        redis = FakeRedis({"lock": "someone-else"})
        heartbeat = LockHeartbeat(redis, "lock", "task-1", clock=FakeClock())
        self.assertFalse(heartbeat.touch())
        self.assertEqual(redis.expired, [])

    def test_bytes_value_is_decoded(self) -> None:
        redis = FakeRedis({"lock": b"task-1"})
        heartbeat = LockHeartbeat(redis, "lock", "task-1", clock=FakeClock())
        self.assertTrue(heartbeat.touch())

    def test_throttle_skips_second_call_within_interval(self) -> None:
        clock = FakeClock()
        redis = FakeRedis({"lock": "task-1"})
        heartbeat = LockHeartbeat(redis, "lock", "task-1", min_interval_seconds=5, clock=clock)
        self.assertTrue(heartbeat.touch())
        clock.now = 1.0
        self.assertFalse(heartbeat.touch())          # 1 秒内被节流
        clock.now = 6.0
        self.assertTrue(heartbeat.touch())           # 超过 5 秒再续

    def test_force_ignores_throttle(self) -> None:
        redis = FakeRedis({"lock": "task-1"})
        heartbeat = LockHeartbeat(redis, "lock", "task-1", clock=FakeClock())
        heartbeat.touch()
        self.assertTrue(heartbeat.touch(force=True))
        self.assertEqual(len(redis.expired), 2)

    def test_missing_client_is_noop(self) -> None:
        heartbeat = LockHeartbeat(None, "lock", "task-1")
        self.assertFalse(heartbeat.touch())

    def test_redis_error_is_swallowed(self) -> None:
        heartbeat = LockHeartbeat(BrokenRedis(), "lock", "task-1", clock=FakeClock())
        self.assertFalse(heartbeat.touch())
        self.assertIn("RuntimeError", heartbeat.last_error or "")


if __name__ == "__main__":
    unittest.main()
