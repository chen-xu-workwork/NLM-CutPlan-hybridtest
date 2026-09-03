"""Concurrent OpenAI-compatible client used by the hybrid planner runtime.

The training repository requester was designed around JSONL batch jobs.  This
module keeps its useful concurrency, rate limiting, and retry behavior while
exposing a request-oriented API suitable for the online search loop.
"""

import asyncio
import hashlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import aiohttp
except ImportError:  # The off-mode baseline intentionally has no LLM deps.
    aiohttp = None


@dataclass(frozen=True)
class LLMClientConfig:
    """Configuration for one shared vLLM/OpenAI-compatible client."""

    base_url: str = "http://127.0.0.1:8091/v1"
    api_key: str = "EMPTY"
    model: str = "Qwen3.5-9B"
    max_concurrency: int = 100
    max_qps: float = 0.0
    max_retries: int = 3
    request_timeout: float = 300.0
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 16384
    seed: Optional[int] = None
    extra_params: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMGenerationResult:
    """Normalized result returned to the bridge request handler."""

    content: Optional[str]
    response: Optional[Dict[str, Any]]
    error: Optional[str]
    attempts: int
    elapsed_seconds: float
    seed: Optional[int] = None

    @property
    def ok(self):
        return self.error is None and self.content is not None


class ReplayLLMRuntime:
    """Deterministic model-runtime substitute for integration testing."""

    def __init__(self, content):
        if not isinstance(content, str) or not content.strip():
            raise ValueError("replay model output must not be empty")
        self.content = content

    def generate(self, messages, request_id=""):
        if not messages:
            raise ValueError("messages must not be empty")
        return LLMGenerationResult(
            content=self.content,
            response=None,
            error=None,
            attempts=1,
            elapsed_seconds=0.0,
        )

    def generate_many(self, messages, count, request_id=""):
        """Return ``count`` deterministic samples for replay integration tests."""

        if count < 1:
            raise ValueError("generation count must be at least 1")
        return tuple(
            self.generate(messages, "%s-sample-%d" % (request_id, index))
            for index in range(count)
        )

    def close(self):
        """Match :class:`BackgroundLLMRuntime`'s lifecycle interface."""


class AsyncRateLimiter:
    """Small token-bucket limiter; a non-positive rate means unlimited."""

    def __init__(self, max_rate, time_period=1.0):
        self.max_rate = float(max_rate)
        self.time_period = float(time_period)
        self.tokens = max(0.0, self.max_rate)
        self.updated_at = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self):
        if self.max_rate <= 0:
            return

        async with self._lock:
            while True:
                now = time.monotonic()
                elapsed = now - self.updated_at
                self.tokens = min(
                    self.max_rate,
                    self.tokens + elapsed * (self.max_rate / self.time_period),
                )
                self.updated_at = now
                if self.tokens >= 1.0:
                    self.tokens -= 1.0
                    return
                wait_time = (
                    (1.0 - self.tokens) * self.time_period / self.max_rate
                )
                await asyncio.sleep(wait_time)


class AsyncLLMClient:
    """Owns one aiohttp session shared by every in-flight state request."""

    RETRYABLE_STATUS = frozenset((429, 500, 502, 503, 504))

    def __init__(self, config):
        self.config = config
        self._semaphore = asyncio.Semaphore(max(1, config.max_concurrency))
        self._rate_limiter = AsyncRateLimiter(config.max_qps)
        self._session = None

    async def start(self):
        """Create the connection pool inside the owning event loop."""

        if aiohttp is None:
            raise RuntimeError("live LLM mode requires the aiohttp package")
        # generate() owns the single end-to-end deadline. A second aiohttp
        # total timeout can intercept wait_for() cancellation and accidentally
        # start another retry after the overall budget has expired.
        timeout = aiohttp.ClientTimeout(total=None)
        connector = aiohttp.TCPConnector(
            limit=max(1, self.config.max_concurrency),
            limit_per_host=max(1, self.config.max_concurrency),
            ttl_dns_cache=300,
        )
        self._session = aiohttp.ClientSession(
            timeout=timeout,
            connector=connector,
        )

    async def close(self):
        """Close persistent HTTP connections."""

        if self._session is not None and not self._session.closed:
            await self._session.close()

    def _headers(self):
        return {
            "Authorization": "Bearer %s" % self.config.api_key,
            "Content-Type": "application/json",
        }

    def _seed_for_request(self, request_id):
        """Derive a stable, distinct seed for each independent sample."""

        if self.config.seed is None:
            return None
        digest = hashlib.sha256(
            ("%d:%s" % (self.config.seed, request_id)).encode("utf-8")
        ).digest()
        return int.from_bytes(digest[:8], "big") % (2 ** 31)

    def _payload(self, messages, request_id=""):
        if not messages:
            raise ValueError("messages must not be empty")
        payload = {
            "model": self.config.model,
            "messages": messages,
            "stream": False,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }
        payload.update(self.config.extra_params)
        request_seed = self._seed_for_request(request_id)
        if request_seed is not None:
            payload["seed"] = request_seed
        return payload

    async def generate(self, messages, request_id=""):
        """Submit one completion within one end-to-end timeout budget."""

        if self._session is None:
            raise RuntimeError("AsyncLLMClient.start() has not been called")
        if not messages:
            raise ValueError("messages must not be empty")

        started_at = time.monotonic()
        request_seed = self._seed_for_request(request_id)
        try:
            return await asyncio.wait_for(
                self._generate_with_retries(
                    messages, started_at, request_id, request_seed
                ),
                timeout=self.config.request_timeout,
            )
        except asyncio.TimeoutError:
            return LLMGenerationResult(
                content=None,
                response=None,
                error="request exceeded total timeout of %.3f seconds"
                % self.config.request_timeout,
                attempts=0,
                elapsed_seconds=time.monotonic() - started_at,
                seed=request_seed,
            )

    async def generate_many(self, messages, count, request_id=""):
        """Generate independent samples concurrently through the shared pool."""

        if count < 1:
            raise ValueError("generation count must be at least 1")
        return tuple(
            await asyncio.gather(
                *[
                    self.generate(
                        messages,
                        "%s-sample-%d" % (request_id, index),
                    )
                    for index in range(count)
                ]
            )
        )

    async def _generate_with_retries(
        self, messages, started_at, request_id, request_seed
    ):
        url = "%s/chat/completions" % self.config.base_url.rstrip("/")
        payload = self._payload(messages, request_id)
        attempts = 0

        for attempt in range(self.config.max_retries + 1):
            attempts = attempt + 1
            try:
                async with self._semaphore:
                    await self._rate_limiter.acquire()
                    async with self._session.post(
                        url,
                        headers=self._headers(),
                        json=payload,
                    ) as response:
                        if response.status == 200:
                            body = await response.json()
                            try:
                                content = body["choices"][0]["message"]["content"]
                            except (KeyError, IndexError, TypeError) as exc:
                                return LLMGenerationResult(
                                    content=None,
                                    response=body,
                                    error="malformed model response: %s" % exc,
                                    attempts=attempts,
                                    elapsed_seconds=time.monotonic() - started_at,
                                    seed=request_seed,
                                )
                            if not isinstance(content, str) or not content.strip():
                                return LLMGenerationResult(
                                    content=None,
                                    response=body,
                                    error="malformed model response: content is empty",
                                    attempts=attempts,
                                    elapsed_seconds=time.monotonic() - started_at,
                                    seed=request_seed,
                                )
                            return LLMGenerationResult(
                                content=content,
                                response=body,
                                error=None,
                                attempts=attempts,
                                elapsed_seconds=time.monotonic() - started_at,
                                seed=request_seed,
                            )

                        error_text = await response.text()
                        if (
                            response.status in self.RETRYABLE_STATUS
                            and attempt < self.config.max_retries
                        ):
                            await asyncio.sleep(2 ** attempt)
                            continue
                        return LLMGenerationResult(
                            content=None,
                            response=None,
                            error="HTTP %d: %s" % (response.status, error_text),
                            attempts=attempts,
                            elapsed_seconds=time.monotonic() - started_at,
                            seed=request_seed,
                        )
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                if attempt < self.config.max_retries:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return LLMGenerationResult(
                    content=None,
                    response=None,
                    error="%s: %s" % (type(exc).__name__, exc),
                    attempts=attempts,
                    elapsed_seconds=time.monotonic() - started_at,
                    seed=request_seed,
                )

        return LLMGenerationResult(
            content=None,
            response=None,
            error="request exhausted retries",
            attempts=attempts,
            elapsed_seconds=time.monotonic() - started_at,
            seed=request_seed,
        )


class BackgroundLLMRuntime:
    """Runs :class:`AsyncLLMClient` on one dedicated asyncio thread.

    The local bridge currently uses ``ThreadingHTTPServer``.  Handler threads
    submit coroutines here and sleep on their futures, so all model traffic
    still shares one non-blocking connection pool rather than creating an
    event loop or HTTP session per state.
    """

    def __init__(self, config):
        self.config = config
        self._loop = None
        self._client = None
        self._thread = None
        self._started = threading.Event()
        self._startup_error = None

    def start(self):
        """Start the event-loop thread and wait for its aiohttp pool."""

        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop,
            name="nlm-llm-client",
            daemon=True,
        )
        self._thread.start()
        self._started.wait()
        if self._startup_error is not None:
            raise RuntimeError(
                "failed to start LLM request runtime: %s" % self._startup_error
            )

    def _run_loop(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._client = AsyncLLMClient(self.config)
        try:
            self._loop.run_until_complete(self._client.start())
        except Exception as exc:
            self._startup_error = exc
            self._started.set()
            self._loop.close()
            return
        self._started.set()
        self._loop.run_forever()
        self._loop.close()

    def generate(self, messages, request_id=""):
        """Synchronously wait for one coroutine from a bridge handler thread."""

        if self._loop is None or self._client is None:
            raise RuntimeError("BackgroundLLMRuntime.start() has not been called")
        future = asyncio.run_coroutine_threadsafe(
            self._client.generate(messages, request_id),
            self._loop,
        )
        return future.result()

    def generate_many(self, messages, count, request_id=""):
        """Run several independent generations concurrently for one state."""

        return self.submit_many(messages, count, request_id).result()

    def submit_many(self, messages, count, request_id=""):
        """Submit a multi-sample state request and expose its cancellable future."""

        if self._loop is None or self._client is None:
            raise RuntimeError("BackgroundLLMRuntime.start() has not been called")
        return asyncio.run_coroutine_threadsafe(
            self._client.generate_many(messages, count, request_id),
            self._loop,
        )

    def submit(self, messages, request_id=""):
        """Submit a request without blocking and return its thread-safe future."""

        if self._loop is None or self._client is None:
            raise RuntimeError("BackgroundLLMRuntime.start() has not been called")
        return asyncio.run_coroutine_threadsafe(
            self._client.generate(messages, request_id),
            self._loop,
        )

    def close(self):
        """Close the shared session and stop the event-loop thread."""

        if self._loop is None or self._thread is None:
            return
        if self._client is not None and self._loop.is_running():
            close_future = asyncio.run_coroutine_threadsafe(
                self._client.close(),
                self._loop,
            )
            try:
                close_future.result(timeout=10)
            except Exception:
                close_future.cancel()
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=10)
        self._thread = None
