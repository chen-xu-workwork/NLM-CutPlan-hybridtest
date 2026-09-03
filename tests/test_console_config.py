import os
import pathlib
import types
import unittest
from unittest import mock

from hybrid_planner.console import (
    DEFAULT_SATISFICING_HEURISTIC,
    DEFAULT_SATISFICING_SEARCH,
    DEFAULT_SEARCH_TIME_LIMIT_SECONDS,
    build_satisficing_search,
    build_single_pass_search,
    build_planner_command,
    configure_planner_environment,
)


class ConsoleEnvironmentTests(unittest.TestCase):
    def test_default_search_is_repeated_satisficing_anytime(self):
        self.assertTrue(DEFAULT_SATISFICING_SEARCH.startswith("iterated([eager("))
        self.assertIn("lmcutnumeric", DEFAULT_SATISFICING_HEURISTIC)
        self.assertIn(
            "tiebreaking([nlm_h, goalcount()])",
            DEFAULT_SATISFICING_SEARCH,
        )
        self.assertIn("llm_h=nlm_h", DEFAULT_SATISFICING_SEARCH)
        self.assertIn("llm_h_open_list_key_index=0", DEFAULT_SATISFICING_SEARCH)
        self.assertIn("pass_bound=true", DEFAULT_SATISFICING_SEARCH)
        self.assertIn("repeat_last=true", DEFAULT_SATISFICING_SEARCH)
        self.assertIn("continue_on_fail=false", DEFAULT_SATISFICING_SEARCH)
        self.assertIn("max_time=7200", DEFAULT_SATISFICING_SEARCH)
        self.assertNotIn("astar(", DEFAULT_SATISFICING_SEARCH)

    def test_search_time_limit_is_applied_at_the_controlling_level(self):
        anytime = build_satisficing_search(17.5)
        self.assertTrue(anytime.startswith("iterated(["))
        self.assertTrue(anytime.endswith("max_time=17.5)"))
        self.assertEqual(anytime.count("max_time="), 1)

        single_pass = build_single_pass_search(9)
        self.assertTrue(single_pass.startswith("eager("))
        self.assertTrue(single_pass.endswith("max_time=9)"))
        self.assertEqual(single_pass.count("max_time="), 1)
        self.assertEqual(DEFAULT_SEARCH_TIME_LIMIT_SECONDS, 7200.0)

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
        self.assertEqual(env["NLM_LLM_RUN_ID"], "problem-1")
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

    def test_off_mode_overrides_every_trigger_switch(self):
        args = types.SimpleNamespace(
            llm_mode="off",
            run_id="baseline-run",
        )
        with mock.patch.dict(
            os.environ,
            {
                "NLM_LLM_TRIGGER": "1",
                "NLM_LLM_REQUEST_INITIAL": "1",
                "NLM_LLM_ENABLE_ANCESTOR_STAGNATION": "1",
                "NLM_LLM_ENABLE_FRONTIER_PLATEAU": "1",
                "NLM_LLM_ENABLE_GLOBAL_STALL": "1",
            },
            clear=True,
        ):
            env = configure_planner_environment(args, "problem-1")

        self.assertEqual(env["NLM_LLM_TRIGGER"], "0")
        self.assertEqual(env["NLM_LLM_REQUEST_INITIAL"], "0")
        self.assertEqual(env["NLM_LLM_ENABLE_ANCESTOR_STAGNATION"], "0")
        self.assertEqual(env["NLM_LLM_ENABLE_FRONTIER_PLATEAU"], "0")
        self.assertEqual(env["NLM_LLM_ENABLE_GLOBAL_STALL"], "0")
        self.assertEqual(env["NLM_LLM_MAX_REQUESTS"], "0")
        self.assertEqual(env["NLM_LLM_RUN_ID"], "baseline-run")

    def test_planner_command_uses_absolute_artifact_paths_and_memory_limit(self):
        args = types.SimpleNamespace(
            python2="python3",
            build="release64",
            plan="plans/sas_plan",
            domain="inputs/domain.pddl",
            problem="inputs/problem.pddl",
            heuristic="h=test()",
            search="astar(h)",
            overall_memory_limit="16G",
        )
        command = build_planner_command(args, pathlib.Path("solver").resolve())

        self.assertIn("--overall-memory-limit", command)
        self.assertEqual(
            command[command.index("--overall-memory-limit") + 1], "16G"
        )
        self.assertTrue(os.path.isabs(command[command.index("--plan-file") + 1]))


if __name__ == "__main__":
    unittest.main()
