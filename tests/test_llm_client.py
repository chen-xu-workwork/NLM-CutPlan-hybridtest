import asyncio
import importlib.util
import unittest

if importlib.util.find_spec("aiohttp") is not None:
    from aiohttp import web
else:
    web = None

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

    def test_returns_three_replay_samples(self):
        runtime = ReplayLLMRuntime("action_Finish(a)")
        results = runtime.generate_many(
            [{"role": "user", "content": "problem"}],
            3,
            request_id="1-0",
        )

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.ok for result in results))
        self.assertEqual(
            [result.content for result in results],
            ["action_Finish(a)"] * 3,
        )

    def test_rejects_empty_replay_output(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            ReplayLLMRuntime("  ")

    def test_sample_seeds_are_stable_and_distinct(self):
        first = AsyncLLMClient(LLMClientConfig(seed=42))
        second = AsyncLLMClient(LLMClientConfig(seed=42))
        identifiers = ["request-sample-%d" % index for index in range(3)]
        first_seeds = [first._seed_for_request(item) for item in identifiers]
        second_seeds = [second._seed_for_request(item) for item in identifiers]

        self.assertEqual(first_seeds, second_seeds)
        self.assertEqual(len(set(first_seeds)), 3)


@unittest.skipUnless(web is not None, "requires aiohttp")
class AsyncLLMClientTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.active = 0
        self.max_active = 0
        self.payloads = []

        async def completions(request):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            payload = await request.json()
            self.payloads.append(payload)
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

    async def test_generate_many_starts_three_samples_concurrently(self):
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
            results = await client.generate_many(
                [{"role": "user", "content": "same-state"}],
                3,
                request_id="state-7",
            )
        finally:
            await client.close()

        self.assertEqual(len(results), 3)
        self.assertTrue(all(result.ok for result in results))
        self.assertEqual(self.max_active, 3)

    async def test_seed_is_reproducible_and_distinct_per_sample(self):
        client = AsyncLLMClient(
            LLMClientConfig(
                base_url="http://127.0.0.1:%d/v1" % self.port,
                seed=42,
                max_concurrency=3,
                max_retries=0,
                request_timeout=5,
            )
        )
        await client.start()
        try:
            results = await client.generate_many(
                [{"role": "user", "content": "seeded"}],
                3,
                request_id="state-7",
            )
        finally:
            await client.close()

        seeds = [result.seed for result in results]
        self.assertEqual(len(set(seeds)), 3)
        self.assertEqual(set(seeds), {payload["seed"] for payload in self.payloads})
        repeated = AsyncLLMClient(LLMClientConfig(seed=42))
        self.assertEqual(
            seeds[0], repeated._seed_for_request("state-7-sample-0")
        )

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
