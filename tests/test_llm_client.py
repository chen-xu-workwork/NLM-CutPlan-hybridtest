import asyncio
import unittest

from aiohttp import web

from hybrid_planner.llm.client import (
    AsyncLLMClient,
    LLMClientConfig,
    ReplayLLMRuntime,
)


class ReplayLLMRuntimeTests(unittest.TestCase):
    def test_returns_the_same_saved_model_output(self):
        runtime = ReplayLLMRuntime("action_Finish(a)")
        result = runtime.generate(
            [{"role": "user", "content": "problem"}],
            request_id="1-0",
        )

        self.assertTrue(result.ok)
        self.assertEqual(result.content, "action_Finish(a)")
        self.assertEqual(result.attempts, 1)

    def test_rejects_empty_replay_output(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            ReplayLLMRuntime("  ")


class AsyncLLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.active = 0
        self.max_active = 0

        async def completions(request):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            payload = await request.json()
            content = payload["messages"][0]["content"]
            delay = 0.2 if content == "return-slow" else 0.03
            try:
                await asyncio.sleep(delay)
            finally:
                self.active -= 1
            if content == "return-empty":
                content = None
            return web.json_response(
                {
                    "choices": [
                        {
                            "message": {
                                "content": content
                            }
                        }
                    ]
                }
            )

        application = web.Application()
        application.router.add_post("/v1/chat/completions", completions)
        self.runner = web.AppRunner(application)
        await self.runner.setup()
        self.site = web.TCPSite(self.runner, "127.0.0.1", 0)
        await self.site.start()
        self.port = self.site._server.sockets[0].getsockname()[1]

    async def asyncTearDown(self):
        await self.runner.cleanup()

    async def test_requests_share_bounded_concurrent_pool(self):
        client = AsyncLLMClient(
            LLMClientConfig(
                base_url="http://127.0.0.1:%d/v1" % self.port,
                max_concurrency=3,
                max_retries=0,
                request_timeout=5,
            )
        )
        await client.start()
        try:
            results = await asyncio.gather(
                *[
                    client.generate(
                        [{"role": "user", "content": "request-%d" % index}]
                    )
                    for index in range(8)
                ]
            )
        finally:
            await client.close()

        self.assertTrue(all(result.ok for result in results))
        self.assertGreater(self.max_active, 1)
        self.assertLessEqual(self.max_active, 3)

    async def test_rejects_empty_model_content(self):
        client = AsyncLLMClient(
            LLMClientConfig(
                base_url="http://127.0.0.1:%d/v1" % self.port,
                max_retries=0,
                request_timeout=5,
            )
        )
        await client.start()
        try:
            result = await client.generate(
                [{"role": "user", "content": "return-empty"}]
            )
        finally:
            await client.close()

        self.assertFalse(result.ok)
        self.assertIn("content is empty", result.error)

    async def test_timeout_is_total_across_retries_and_queueing(self):
        client = AsyncLLMClient(
            LLMClientConfig(
                base_url="http://127.0.0.1:%d/v1" % self.port,
                max_retries=3,
                request_timeout=0.05,
            )
        )
        await client.start()
        started_at = asyncio.get_running_loop().time()
        try:
            result = await client.generate(
                [{"role": "user", "content": "return-slow"}]
            )
        finally:
            await client.close()
        elapsed = asyncio.get_running_loop().time() - started_at

        self.assertFalse(result.ok)
        self.assertIn("total timeout", result.error)
        self.assertLess(elapsed, 0.15)


if __name__ == "__main__":
    unittest.main()
