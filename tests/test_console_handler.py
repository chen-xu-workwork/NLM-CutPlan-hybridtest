import importlib.util
import json
import pathlib
import tempfile
import threading
import unittest
import urllib.request
from dataclasses import dataclass
from http.server import ThreadingHTTPServer

from hybrid_planner.console import make_handler
from hybrid_planner.llm.client import (
    LLMGenerationResult,
    ReplayLLMRuntime,
)
from hybrid_planner.prompting.builder import (
    HybridPromptBuilder,
    PromptBuilderConfig,
)
from hybrid_planner.validation.response_processor import (
    PlanResponseProcessor,
    ProcessedModelResponse,
    UnifiedPlanningPrefixValidator,
)


HAS_PLANNING_RUNTIME = (
    importlib.util.find_spec("pddl") is not None
    and importlib.util.find_spec("unified_planning") is not None
)


@dataclass
class FakePrompts:
    problem_path = type("ProblemPath", (), {"name": "problem.pddl"})()
    system: str = "system"
    user: str = "user"
    problem_description: str = "description"
    runtime_problem: str = "(define (problem runtime))"

    def as_messages(self):
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]


class FakePromptBuilder:
    def build(self, problem_id, init_text):
        return FakePrompts()


class FakeRuntime:
    def generate(self, messages, request_id=""):
        return LLMGenerationResult(
            content="action_Drive(truck0, depot0, distributor0)",
            response={"choices": []},
            error=None,
            attempts=1,
            elapsed_seconds=0.25,
        )

    def generate_many(self, messages, count, request_id=""):
        return tuple(
            self.generate(messages, "%s-sample-%d" % (request_id, index))
            for index in range(count)
        )


class FakeProcessor:
    def process(self, generated_text, runtime_problem_text):
        return ProcessedModelResponse(
            status="ok",
            actions=("(drive truck0 depot0 distributor0)",),
            generated_action_count=1,
            legal_action_count=1,
            goal_reached=False,
        )


class ConsoleHandlerTests(unittest.TestCase):
    @staticmethod
    def post_json(server, payload):
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            "http://127.0.0.1:%d/llm/request"
            % server.server_address[1],
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read().decode("utf-8"))

    def test_live_handler_returns_processed_legal_actions(self):
        server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            make_handler(
                "/llm/request",
                FakePromptBuilder(),
                llm_runtime=FakeRuntime(),
                response_processor=FakeProcessor(),
            ),
        )
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = self.post_json(
                server,
                {
                    "request_id": "7-1",
                    "state_id": 7,
                    "problem_id": "problem",
                    "init": "(:init)",
                },
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(
            result["actions"],
            ["(drive truck0 depot0 distributor0)"],
        )
        self.assertEqual(result["sample_count"], 3)
        self.assertEqual(result["usable_sample_count"], 3)
        self.assertEqual(
            result["action_chains"],
            [["(drive truck0 depot0 distributor0)"]] * 3,
        )
        self.assertEqual(result["request_id"], "7-1")

    @unittest.skipUnless(
        HAS_PLANNING_RUNTIME,
        "requires pddl and unified-planning",
    )
    def test_real_prompt_and_validation_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = pathlib.Path(temp_dir)
            problem_dir = root / "problems"
            problem_dir.mkdir()
            domain_path = root / "domain.pddl"
            domain_path.write_text(
                "(define (domain demo)\n"
                "  (:requirements :strips :typing)\n"
                "  (:types item)\n"
                "  (:predicates (ready ?x - item) (done ?x - item))\n"
                "  (:action Finish\n"
                "    :parameters (?x - item)\n"
                "    :precondition (ready ?x)\n"
                "    :effect (and (not (ready ?x)) (done ?x)))\n"
                ")\n",
                encoding="utf-8",
            )
            problem_path = problem_dir / "problem-1.pddl"
            problem_path.write_text(
                "(define (problem p1)\n"
                "  (:domain demo)\n"
                "  (:objects a - item)\n"
                "  (:init (ready a))\n"
                "  (:goal (done a))\n"
                ")\n",
                encoding="utf-8",
            )
            domain_code = root / "domain.txt"
            domain_code.write_text("demo domain", encoding="utf-8")

            prompt_builder = HybridPromptBuilder(
                PromptBuilderConfig(
                    domain_pddl=domain_path,
                    problem_dir=problem_dir,
                    domain_code=domain_code,
                )
            )
            processor = PlanResponseProcessor(
                UnifiedPlanningPrefixValidator(domain_path)
            )
            server = ThreadingHTTPServer(
                ("127.0.0.1", 0),
                make_handler(
                    "/llm/request",
                    prompt_builder,
                    llm_runtime=ReplayLLMRuntime("action_Finish(a)"),
                    response_processor=processor,
                ),
            )
            thread = threading.Thread(
                target=server.serve_forever,
                daemon=True,
            )
            thread.start()
            try:
                result = self.post_json(
                    server,
                    {
                        "request_id": "1-0",
                        "state_id": 1,
                        "problem_id": "problem-1",
                        "init": "(:init (ready a))",
                    },
                )
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["actions"], ["(finish a)"])
        self.assertEqual(result["action_chains"], [["(finish a)"]] * 3)
        self.assertEqual(result["legal_action_count"], 1)
        self.assertTrue(result["goal_reached"])


if __name__ == "__main__":
    unittest.main()
