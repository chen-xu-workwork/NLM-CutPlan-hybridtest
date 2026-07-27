"""Parse model output and validate its longest executable action prefix."""

import ast
import pathlib
import re
import tempfile
import threading
from dataclasses import dataclass


class PlanParseError(ValueError):
    """Raised when model output cannot be converted into action calls."""


class PlanValidationError(RuntimeError):
    """Raised when the validation environment or PDDL cannot be loaded."""


@dataclass(frozen=True)
class ActionCall:
    """One action emitted by the model in its training-time Python syntax."""

    python_name: str
    pddl_name: str
    parameters: tuple
    line_number: int

    def as_python(self):
        return "%s(%s)" % (self.python_name, ", ".join(self.parameters))

    def as_pddl(self):
        suffix = " " + " ".join(self.parameters) if self.parameters else ""
        return "(%s%s)" % (self.pddl_name, suffix)


@dataclass(frozen=True)
class PrefixValidationResult:
    """Result of simulating an action sequence from the runtime initial state."""

    legal_actions: tuple
    total_actions: int
    invalid_action_index: object = None
    error: object = None
    goal_reached: object = None


@dataclass(frozen=True)
class ProcessedModelResponse:
    """Compact application-level response returned to the C++ bridge."""

    status: str
    actions: tuple
    generated_action_count: int
    legal_action_count: int
    invalid_action_index: object = None
    error: object = None
    goal_reached: object = None

    def as_dict(self):
        result = {
            "status": self.status,
            "actions": list(self.actions),
            "generated_action_count": self.generated_action_count,
            "legal_action_count": self.legal_action_count,
        }
        if self.invalid_action_index is not None:
            result["invalid_action_index"] = self.invalid_action_index
        if self.error:
            result["error"] = self.error
        if self.goal_reached is not None:
            result["goal_reached"] = self.goal_reached
        return result


def extract_python_code(text):
    """Return the longest fenced code block, or the complete response."""

    if not text or not text.strip():
        raise PlanParseError("model response is empty")
    matches = re.findall(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if matches:
        return max(matches, key=len).strip()
    return text.strip()


def _python_action_to_pddl(python_name):
    raw_name = python_name[len("action_") :]
    with_boundaries = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "-", raw_name)
    return with_boundaries.replace("_", "-").lower()


def _is_action_call(node):
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id.startswith("action_")
    )


def _parse_action_call(node):
    parameters = []
    for argument in node.args:
        if isinstance(argument, ast.Name):
            parameters.append(argument.id)
        elif isinstance(argument, ast.Constant) and isinstance(
            argument.value, str
        ):
            parameters.append(argument.value)
        else:
            raise PlanParseError(
                "unsupported parameter in %s at line %d"
                % (node.func.id, getattr(node, "lineno", 0))
            )
    if node.keywords:
        raise PlanParseError(
            "keyword arguments are not supported in %s at line %d"
            % (node.func.id, getattr(node, "lineno", 0))
        )
    return ActionCall(
        python_name=node.func.id,
        pddl_name=_python_action_to_pddl(node.func.id),
        parameters=tuple(parameters),
        line_number=getattr(node, "lineno", 0),
    )


def extract_action_calls(text):
    """Safely parse ordered ``action_Xxx(...)`` calls without executing code."""

    code = extract_python_code(text)
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        raise PlanParseError(
            "generated Python is invalid at line %s: %s"
            % (exc.lineno, exc.msg)
        ) from exc
    actions = []
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and _is_action_call(statement.value):
            actions.append(_parse_action_call(statement.value))
            continue
        nested_actions = [
            node for node in ast.walk(statement) if _is_action_call(node)
        ]
        if nested_actions:
            first = nested_actions[0]
            raise PlanParseError(
                "action calls must be top-level sequential statements; "
                "%s is nested at line %d"
                % (first.func.id, getattr(first, "lineno", 0))
            )
    if not actions:
        raise PlanParseError("no action_Xxx(...) calls were found")
    return actions


class UnifiedPlanningPrefixValidator:
    """Uses Unified Planning to keep only the longest legal action prefix."""

    def __init__(self, domain_pddl):
        self.domain_pddl = pathlib.Path(domain_pddl)

    @staticmethod
    def validate_environment():
        """Fail early when the runtime environment lacks Unified Planning."""

        try:
            import unified_planning  # noqa: F401
        except ImportError as exc:
            raise PlanValidationError(
                "unified-planning is required in live mode; "
                "install requirements/hybrid.txt in the WSL environment"
            ) from exc

    def validate(self, runtime_problem_text, actions):
        """Simulate actions in order and stop at the first invalid action."""

        self.validate_environment()
        try:
            from unified_planning.io import PDDLReader
            from unified_planning.shortcuts import SequentialSimulator
        except ImportError as exc:
            raise PlanValidationError(str(exc)) from exc

        try:
            with tempfile.TemporaryDirectory(prefix="nlm_validate_") as temp_dir:
                problem_path = pathlib.Path(temp_dir) / "runtime_problem.pddl"
                problem_path.write_text(runtime_problem_text, encoding="utf-8")
                problem = PDDLReader().parse_problem(
                    str(self.domain_pddl),
                    str(problem_path),
                )
        except Exception as exc:
            raise PlanValidationError("failed to parse runtime PDDL: %s" % exc) from exc

        legal_actions = []
        invalid_index = None
        invalid_error = None
        goal_reached = None

        try:
            with SequentialSimulator(problem=problem) as simulator:
                current_state = simulator.get_initial_state()
                is_goal = getattr(simulator, "is_goal", None)
                if callable(is_goal):
                    goal_reached = bool(is_goal(current_state))
                for index, action_call in enumerate(actions):
                    if goal_reached:
                        break
                    try:
                        action_definition = problem.action(action_call.pddl_name)
                        parameters = tuple(
                            problem.object(name) for name in action_call.parameters
                        )
                    except ValueError as exc:
                        invalid_index = index
                        invalid_error = "unknown action or object: %s" % exc
                        break

                    try:
                        applicable = simulator.is_applicable(
                            current_state,
                            action_definition,
                            parameters,
                        )
                    except TypeError:
                        # Compatibility with UP versions preferring ActionInstance.
                        from unified_planning.plans import ActionInstance

                        action_instance = ActionInstance(
                            action_definition,
                            parameters,
                        )
                        applicable = simulator.is_applicable(
                            current_state,
                            action_instance,
                        )

                    if not applicable:
                        invalid_index = index
                        invalid_error = "action is not applicable: %s" % (
                            action_call.as_pddl(),
                        )
                        break

                    try:
                        current_state = simulator.apply(
                            current_state,
                            action_definition,
                            parameters,
                        )
                    except TypeError:
                        from unified_planning.plans import ActionInstance

                        current_state = simulator.apply(
                            current_state,
                            ActionInstance(action_definition, parameters),
                        )
                    legal_actions.append(action_call)

                    if callable(is_goal):
                        goal_reached = bool(is_goal(current_state))
        except PlanValidationError:
            raise
        except Exception as exc:
            raise PlanValidationError("simulation failed: %s" % exc) from exc

        return PrefixValidationResult(
            legal_actions=tuple(legal_actions),
            total_actions=len(actions),
            invalid_action_index=invalid_index,
            error=invalid_error,
            goal_reached=goal_reached,
        )


class PlanResponseProcessor:
    """Combines extraction and validation into the bridge response contract."""

    def __init__(self, validator, max_validation_concurrency=4):
        self.validator = validator
        self._validation_slots = threading.BoundedSemaphore(
            max(1, int(max_validation_concurrency))
        )

    def process(self, generated_text, runtime_problem_text):
        try:
            action_calls = extract_action_calls(generated_text)
        except PlanParseError as exc:
            return ProcessedModelResponse(
                status="parse_error",
                actions=(),
                generated_action_count=0,
                legal_action_count=0,
                error=str(exc),
            )

        try:
            with self._validation_slots:
                validation = self.validator.validate(
                    runtime_problem_text,
                    action_calls,
                )
        except PlanValidationError as exc:
            return ProcessedModelResponse(
                status="validation_error",
                actions=(),
                generated_action_count=len(action_calls),
                legal_action_count=0,
                error=str(exc),
            )

        legal_pddl = tuple(action.as_pddl() for action in validation.legal_actions)
        if not legal_pddl:
            status = "invalid_plan"
        elif validation.goal_reached:
            status = "ok"
        elif len(legal_pddl) < len(action_calls):
            status = "partial"
        else:
            status = "ok"
        return ProcessedModelResponse(
            status=status,
            actions=legal_pddl,
            generated_action_count=len(action_calls),
            legal_action_count=len(legal_pddl),
            invalid_action_index=validation.invalid_action_index,
            error=validation.error,
            goal_reached=validation.goal_reached,
        )
