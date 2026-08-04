import json
import os
from pathlib import Path
import select
import subprocess
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = PROJECT_ROOT / "builds" / "release64" / "bin" / "nlm-score"
# `output.sas` is the translator result. The C++ search/scorer consumes the
# preprocessor result, which this repository keeps as `output`.
TASK_PATH = PROJECT_ROOT / "output"
TASK_HASH = "test:depots-preprocessed-output"

# The first twelve actions reach the checked-in fixture's goal. The final
# unknown action verifies that text emitted after a valid plan is ignored.
GOAL_PLAN_WITH_TRAILING_ACTION = [
    "(lift hoist0 crate1 crate3 depot0)",
    "(load hoist0 crate1 truck0 depot0)",
    "(lift hoist0 crate3 crate4 depot0)",
    "(load hoist0 crate3 truck0 depot0)",
    "(lift hoist0 crate4 pallet1 depot0)",
    "(load hoist0 crate4 truck0 depot0)",
    "(unload hoist0 crate3 truck0 depot0)",
    "(drop hoist0 crate3 pallet1 depot0)",
    "(unload hoist0 crate1 truck0 depot0)",
    "(drop hoist0 crate1 crate3 depot0)",
    "(lift hoist0 crate5 pallet3 depot0)",
    "(drop hoist0 crate5 pallet2 depot0)",
    "(does-not-exist)",
]

STATE_FIELDS = {"state_index", "state_id", "h"}
REMOVED_RESPONSE_FIELDS = {
    "valid_prefix_length",
    "stop_reason",
    "invalid_action",
    "trailing_action_count",
    "goal_reached",
    "goal_state_index",
    "start_h",
    "final_h",
    "best_h",
    "best_h_state_index",
    "unique_state_count",
    "repeated_state_count",
    "first_repeat_action_index",
    "dead_end_reached",
}


def _binary_path():
    configured = os.environ.get("NLM_SCORE_BIN")
    return Path(configured).resolve() if configured else DEFAULT_BINARY


def _read_json_line(stream, timeout=30):
    readable, _, _ = select.select([stream], [], [], timeout)
    if not readable:
        raise TimeoutError("timed out waiting for nlm-score JSONL output")
    line = stream.readline()
    if not line:
        raise EOFError("nlm-score closed stdout before returning a message")
    return json.loads(line)


@unittest.skipUnless(os.name == "posix", "nlm-score is built and run in Linux/WSL")
class TrajectoryScorerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.binary = _binary_path()
        if not cls.binary.is_file() or not os.access(cls.binary, os.X_OK):
            raise unittest.SkipTest(
                "build nlm-score first or set NLM_SCORE_BIN to the executable"
            )
        if not TASK_PATH.is_file():
            raise unittest.SkipTest("preprocessed output integration fixture is missing")

    def _command(self, mode, *extra):
        return [
            str(self.binary),
            "--task",
            str(TASK_PATH),
            "--task-hash",
            TASK_HASH,
            mode,
            "--max-score-seconds",
            "30",
            *extra,
        ]

    @staticmethod
    def _request(request_id, actions):
        return {
            "type": "score_request",
            "protocol_version": 1,
            "request_id": request_id,
            "problem_id": "depots-preprocessed-output",
            "task_hash": TASK_HASH,
            "actions": actions,
        }

    def _assert_planning_response(
        self, response, outcome, generated_count, applied_count, invalid_index=None
    ):
        self.assertEqual(response["type"], "score_response")
        self.assertEqual(response["status"], "ok")
        self.assertEqual(response["outcome"], outcome)
        self.assertEqual(response["generated_action_count"], generated_count)
        self.assertEqual(response["applied_action_count"], applied_count)
        self.assertEqual(response["invalid_action_index"], invalid_index)
        self.assertEqual(len(response["states"]), applied_count + 1)
        self.assertTrue(REMOVED_RESPONSE_FIELDS.isdisjoint(response))

        for expected_index, state in enumerate(response["states"]):
            self.assertEqual(set(state), STATE_FIELDS)
            self.assertEqual(state["state_index"], expected_index)
            self.assertIsInstance(state["state_id"], int)
            self.assertTrue(
                state["h"] is None or isinstance(state["h"], (int, float))
            )

    def test_once_returns_legal_incomplete_with_initial_state(self):
        request = self._request("once-zero", [])
        completed = subprocess.run(
            self._command("--once"),
            input=json.dumps(request) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=True,
        )
        messages = [json.loads(line) for line in completed.stdout.splitlines()]
        self.assertEqual(len(messages), 1)
        response = messages[0]
        self.assertEqual(response["request_id"], "once-zero")
        self._assert_planning_response(
            response, "legal_incomplete", generated_count=0, applied_count=0
        )

    def test_once_goal_reached_ignores_trailing_action(self):
        request = self._request("goal-with-trailing", GOAL_PLAN_WITH_TRAILING_ACTION)
        completed = subprocess.run(
            self._command("--once"),
            input=json.dumps(request) + "\n",
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=60,
            check=True,
        )
        response = json.loads(completed.stdout)
        self._assert_planning_response(
            response,
            "goal_reached",
            generated_count=len(GOAL_PLAN_WITH_TRAILING_ACTION),
            applied_count=len(GOAL_PLAN_WITH_TRAILING_ACTION) - 1,
        )

    def test_stream_returns_legal_prefix_recovers_and_shuts_down(self):
        process = subprocess.Popen(
            self._command("--stream"),
            text=True,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=1,
        )
        self.addCleanup(self._stop_process, process)

        ready = _read_json_line(process.stdout)
        self.assertEqual(ready["type"], "ready")
        self.assertEqual(ready["protocol_version"], 1)
        self.assertEqual(ready["task_hash"], TASK_HASH)
        self.assertNotIn("max_actions", ready)

        malformed = {"type": "score_request", "protocol_version": 1}
        self._send(process, malformed)
        invalid_request = _read_json_line(process.stdout)
        self.assertEqual(invalid_request["status"], "invalid_request")
        self.assertIsNone(invalid_request["outcome"])
        self.assertEqual(invalid_request["states"], [])

        mismatch = self._request("mismatch", [])
        mismatch["task_hash"] = "wrong-task"
        self._send(process, mismatch)
        mismatch_response = _read_json_line(process.stdout)
        self.assertEqual(mismatch_response["status"], "task_mismatch")
        self.assertIsNone(mismatch_response["outcome"])

        invalid_after_prefix = [
            "(lift hoist0 crate1 crate3 depot0)",
            "(does-not-exist)",
            "(load hoist0 crate1 truck0 depot0)",
        ]
        self._send(
            process,
            self._request("invalid-after-prefix", invalid_after_prefix),
        )
        invalid = _read_json_line(process.stdout)
        self._assert_planning_response(
            invalid,
            "invalid",
            generated_count=3,
            applied_count=1,
            invalid_index=1,
        )

        self._send(
            process,
            self._request(
                "inapplicable",
                ["(load hoist0 crate1 truck0 depot0)"],
            ),
        )
        inapplicable = _read_json_line(process.stdout)
        self._assert_planning_response(
            inapplicable,
            "invalid",
            generated_count=1,
            applied_count=0,
            invalid_index=0,
        )

        legal_actions = [
            "(lift hoist0 crate1 crate3 depot0)",
            "(load hoist0 crate1 truck0 depot0)",
        ]
        self._send(process, self._request("legal-1", legal_actions))
        first = _read_json_line(process.stdout)
        self._send(process, self._request("legal-2", legal_actions))
        second = _read_json_line(process.stdout)

        self._assert_planning_response(
            first, "legal_incomplete", generated_count=2, applied_count=2
        )
        self._assert_planning_response(
            second, "legal_incomplete", generated_count=2, applied_count=2
        )
        self.assertEqual(first["path_cost"], second["path_cost"])
        self.assertEqual(first["states"], second["states"])

        self._send(
            process,
            {
                "type": "shutdown",
                "protocol_version": 1,
                "request_id": "shutdown-test",
            },
        )
        shutdown = _read_json_line(process.stdout)
        self.assertEqual(shutdown["type"], "shutdown_ack")
        self.assertEqual(shutdown["request_id"], "shutdown-test")
        self.assertEqual(process.wait(timeout=10), 0)
        self.assertEqual(
            process.stdout.read(),
            "",
            "stdout must contain JSONL protocol messages only",
        )

    @staticmethod
    def _send(process, message):
        process.stdin.write(json.dumps(message) + "\n")
        process.stdin.flush()

    @staticmethod
    def _stop_process(process):
        try:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
        finally:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


if __name__ == "__main__":
    unittest.main()
