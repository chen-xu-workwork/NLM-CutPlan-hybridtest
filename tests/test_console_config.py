import os
import types
import unittest
from unittest import mock

from hybrid_planner.console import configure_planner_environment


class ConsoleEnvironmentTests(unittest.TestCase):
    def test_live_defaults_are_bounded_and_conservative(self):
        args = types.SimpleNamespace(
            host="127.0.0.1",
            actual_port=8765,
            path="/llm/request",
            pending_behavior="skip",
            emit_state="0",
            llm_timeout=300.0,
            http_workers=0,
            llm_max_concurrency=100,
            llm_mode="live",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            env = configure_planner_environment(args, "problem-1")

        self.assertEqual(env["NLM_LLM_HTTP_WORKERS"], "100")
        self.assertEqual(env["NLM_LLM_MAX_PENDING"], "100")
        self.assertEqual(env["NLM_LLM_CHECK_INTERVAL"], "50")
        self.assertEqual(env["NLM_LLM_STALL_EXPANSIONS"], "500")
        self.assertEqual(env["NLM_LLM_H_RELATIVE_EPSILON"], "0.01")
        self.assertEqual(env["NLM_LLM_PLATEAU_H_CV"], "0.05")
        self.assertEqual(env["NLM_LLM_PLATEAU_F_CV"], "0.05")

    def test_existing_trigger_overrides_are_preserved(self):
        args = types.SimpleNamespace(
            host="127.0.0.1",
            actual_port=8765,
            path="/llm/request",
            pending_behavior="normal",
            emit_state="0",
            llm_timeout=120.0,
            http_workers=12,
            llm_max_concurrency=100,
            llm_mode="live",
        )
        with mock.patch.dict(
            os.environ,
            {
                "NLM_LLM_MAX_PENDING": "7",
                "NLM_LLM_CHECK_INTERVAL": "9",
            },
            clear=True,
        ):
            env = configure_planner_environment(args, "problem-1")

        self.assertEqual(env["NLM_LLM_HTTP_WORKERS"], "12")
        self.assertEqual(env["NLM_LLM_MAX_PENDING"], "7")
        self.assertEqual(env["NLM_LLM_CHECK_INTERVAL"], "9")


if __name__ == "__main__":
    unittest.main()
