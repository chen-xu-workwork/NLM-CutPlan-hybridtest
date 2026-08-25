import os
import types
import unittest
from unittest import mock

from hybrid_planner.console import (
    DEFAULT_SATISFICING_SEARCH,
    configure_planner_environment,
)


class ConsoleEnvironmentTests(unittest.TestCase):
    def test_default_search_is_eager_satisficing(self):
        self.assertTrue(DEFAULT_SATISFICING_SEARCH.startswith("eager_greedy("))
        self.assertIn("lmcutnumeric", DEFAULT_SATISFICING_SEARCH)
        self.assertNotIn("astar(", DEFAULT_SATISFICING_SEARCH)

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
            llm_samples_per_state=3,
            llm_mode="live",
        )
        with mock.patch.dict(os.environ, {}, clear=True):
            env = configure_planner_environment(args, "problem-1")

        self.assertEqual(env["NLM_LLM_HTTP_WORKERS"], "100")
        self.assertEqual(env["NLM_LLM_MAX_PENDING"], "33")
        self.assertEqual(env["NLM_LLM_MAX_REQUESTS"], "10")
        self.assertEqual(env["NLM_LLM_ANALYSIS_INTERVAL"], "8192")
        self.assertEqual(env["NLM_LLM_ACTIVITY_WINDOWS"], "4")
        self.assertEqual(env["NLM_LLM_GROWTH_CONFIRM_WINDOWS"], "2")
        self.assertEqual(env["NLM_LLM_LAYER_RESET_WINDOWS"], "4")
        self.assertEqual(
            env["NLM_LLM_LAYER_MIN_RECENT_EXPANDED"], "4096"
        )
        self.assertEqual(
            env["NLM_LLM_LAYER_MIN_RECENT_NET_GROWTH"], "1024"
        )
        self.assertEqual(
            env["NLM_LLM_LAYER_MIN_SINCE_REQUEST_EXPANDED"], "8192"
        )
        self.assertEqual(
            env["NLM_LLM_LAYER_MIN_SINCE_REQUEST_NET_GROWTH"], "2048"
        )
        self.assertEqual(env["NLM_LLM_PLATEAU_GROWTH_RATIO"], "1.05")
        self.assertEqual(env["NLM_LLM_STALL_EXPANSIONS"], "500000")
        self.assertEqual(env["NLM_LLM_ANCESTOR_CHECK_INTERVAL"], "100000")
        self.assertEqual(env["NLM_LLM_ANCESTOR_DEPTH"], "10")
        self.assertEqual(env["NLM_LLM_MIN_DEPTH"], "20")
        self.assertEqual(
            env["NLM_LLM_MIN_REQUEST_GAP_EXPANSIONS"], "100000"
        )
        self.assertEqual(
            env["NLM_LLM_PER_LAYER_REQUEST_GAP_EXPANSIONS"], "500000"
        )
        self.assertEqual(env["NLM_LLM_CANDIDATE_LAYERS"], "3")
        self.assertEqual(env["NLM_LLM_REQUESTS_PER_SLOT"], "1")
        self.assertEqual(env["NLM_LLM_HEARTBEAT_INTERVAL"], "100000")
        self.assertEqual(env["NLM_LLM_H_RELATIVE_EPSILON"], "0.005")

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
            llm_samples_per_state=3,
            llm_mode="live",
        )
        with mock.patch.dict(
            os.environ,
            {
                "NLM_LLM_MAX_PENDING": "7",
                "NLM_LLM_MAX_REQUESTS": "5",
                "NLM_LLM_ANALYSIS_INTERVAL": "9",
            },
            clear=True,
        ):
            env = configure_planner_environment(args, "problem-1")

        self.assertEqual(env["NLM_LLM_HTTP_WORKERS"], "12")
        self.assertEqual(env["NLM_LLM_MAX_PENDING"], "7")
        self.assertEqual(env["NLM_LLM_MAX_REQUESTS"], "5")
        self.assertEqual(env["NLM_LLM_ANALYSIS_INTERVAL"], "9")


if __name__ == "__main__":
    unittest.main()
