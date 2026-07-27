import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
SAS_TASKS_PATH = ROOT / "src" / "translate" / "sas_tasks.py"


def load_sas_tasks():
    spec = importlib.util.spec_from_file_location(
        "nlm_translate_sas_tasks",
        str(SAS_TASKS_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAtom:
    def __init__(self, predicate, *args):
        self.predicate = predicate
        self.args = args


class FakeFluent:
    def __init__(self, symbol, *args):
        self.symbol = symbol
        self.args = args


class FakeNumber:
    def __init__(self, value):
        self.value = value


class FakeAssignment:
    def __init__(self, symbol, args, value):
        self.fluent = FakeFluent(symbol, *args)
        self.expression = FakeNumber(value)


class InitConstantSerializationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sas_tasks = load_sas_tasks()

    def test_object_equality_atoms_are_not_exported(self):
        facts = self.sas_tasks._constant_init_fact_strings(
            [
                FakeAtom("=", "crate0", "crate0"),
                FakeAtom("at", "hoist0", "depot0"),
            ],
            [],
        )

        self.assertEqual(facts, ["(at hoist0 depot0)"])

    def test_static_numeric_assignments_keep_pddl_function_syntax(self):
        facts = self.sas_tasks._constant_init_fact_strings(
            [],
            [
                FakeAssignment("weight", ("crate0",), 90.0),
                FakeAssignment("load_limit", ("truck0",), 506.0),
            ],
        )

        self.assertEqual(
            facts,
            [
                "(= (load_limit truck0) 506.0)",
                "(= (weight crate0) 90.0)",
            ],
        )


if __name__ == "__main__":
    unittest.main()
