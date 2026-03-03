from app import main
from app.schemas import ChatResponse


class FakeRedis:
    def __init__(self):
        self.store = {}

    def ping(self):
        return True

    def get(self, key):
        return self.store.get(key)

    def setex(self, key, ttl, value):
        self.store[key] = value
        return True

    def scan_iter(self, match=None):
        if not match:
            for key in self.store:
                yield key
            return
        prefix = match.rstrip("*")
        for key in self.store:
            if key.startswith(prefix):
                yield key


def test_redis_cache_get_put_and_stats():
    original_backend = main.CACHE_BACKEND
    original_client = main.REDIS_CLIENT
    original_prefix = main.REDIS_KEY_PREFIX

    try:
        main.CACHE_BACKEND = "redis"
        main.REDIS_KEY_PREFIX = "test:chat:"
        main.REDIS_CLIENT = FakeRedis()
        main.REQUEST_ID.set("rid-cache")

        response = ChatResponse(
            response="cached text",
            latency_ms=1,
            prompt_tokens=1,
            completion_tokens=2,
            model="m",
            request_id="rid-orig",
            context_window=2048,
            context_used_pct=0.1,
            model_type="decoder-only",
            attention_masking="causal",
            attention_heads=1,
            hidden_size=1,
            estimated_attention_ops=9,
            total_tokens=3,
            output_to_input_ratio=2.0,
            refined=False,
            original_response=None,
            critique=None,
            refine_steps_used=0,
            cache_hit=False,
        )

        main._put_cached_chat_response("k", response)
        cached = main._get_cached_chat_response("k")

        assert cached is not None
        assert cached.response == "cached text"
        assert cached.cache_hit is True
        assert cached.request_id == "rid-cache"

        stats = main._cache_stats()
        assert stats["backend"] == "redis"
        assert stats["entries"] == 1
    finally:
        main.CACHE_BACKEND = original_backend
        main.REDIS_CLIENT = original_client
        main.REDIS_KEY_PREFIX = original_prefix
