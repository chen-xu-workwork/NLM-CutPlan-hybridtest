import unittest
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from hybrid_planner.validation.response_processor import (
    PlanParseError,
    PlanResponseProcessor,
    PrefixValidationResult,
    extract_action_calls,
)


class FakeValidator:
    def __init__(self, legal_count, error=None):
        self.legal_count = legal_count
        self.error = error

    def validate(self, runtime_problem_text, actions):
        return PrefixValidationResult(
            legal_actions=tuple(actions[: self.legal_count]),
            total_actions=len(actions),
            invalid_action_index=(
                self.legal_count if self.legal_count < len(actions) else None
            ),
            error=self.error,
            goal_reached=False,
        )


class ConcurrentValidator(FakeValidator):
    def __init__(self):
        super().__init__(legal_count=1)
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def validate(self, runtime_problem_text, actions):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(0.03)
        with self.lock:
            self.active -= 1
        return super().validate(runtime_problem_text, actions)


class ActionExtractionTests(unittest.TestCase):
    def test_extracts_actions_from_training_format(self):
        output = """
<think>reasoning</think>
```python
assert is_clear(crate1)
action_Lift(hoist0, crate1, pallet0, depot0)
action_Load(hoist0, crate1, truck0, depot0)
```
"""
        actions = extract_action_calls(output)
        self.assertEqual(
            [action.as_pddl() for action in actions],
            [
                "(lift hoist0 crate1 pallet0 depot0)",
                "(load hoist0 crate1 truck0 depot0)",
            ],
        )

    def test_converts_pascal_case_to_hyphenated_pddl(self):
        actions = extract_action_calls("action_LoadTruck(crate0, truck0)")
        self.assertEqual(actions[0].pddl_name, "load-truck")

    def test_rejects_computed_parameters(self):
        with self.assertRaises(PlanParseError):
            extract_action_calls("action_Load(get_crate(), truck0)")

    def test_rejects_branching_action_calls(self):
        with self.assertRaises(PlanParseError):
            extract_action_calls(
                "if is_clear(crate0):\n"
                "    action_Lift(hoist0, crate0, pallet0, depot0)"
            )


class ResponseProcessorTests(unittest.TestCase):
    def test_returns_only_legal_prefix(self):
        processor = PlanResponseProcessor(
            FakeValidator(legal_count=1, error="second action is invalid")
        )
        processed = processor.process(
            """
```python
action_Lift(hoist0, crate1, pallet0, depot0)
action_Load(hoist0, crate1, truck0, depot0)
```
""",
            "(define (problem placeholder))",
        )
        self.assertEqual(processed.status, "partial")
        self.assertEqual(
            processed.actions,
            ("(lift hoist0 crate1 pallet0 depot0)",),
        )
        self.assertEqual(processed.invalid_action_index, 1)

    def test_limits_cpu_side_validation_concurrency(self):
        validator = ConcurrentValidator()
        processor = PlanResponseProcessor(
            validator,
            max_validation_concurrency=2,
        )
        with ThreadPoolExecutor(max_workers=6) as executor:
            results = list(
                executor.map(
                    lambda _: processor.process(
                        "action_Drive(truck0, depot0, distributor0)",
                        "(define (problem placeholder))",
                    ),
                    range(6),
                )
            )
        self.assertTrue(all(result.status == "ok" for result in results))
        self.assertGreater(validator.max_active, 1)
        self.assertLessEqual(validator.max_active, 2)


if __name__ == "__main__":
    unittest.main()
