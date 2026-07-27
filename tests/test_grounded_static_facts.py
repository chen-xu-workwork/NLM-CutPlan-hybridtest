import importlib.util
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "src" / "translate" / "grounded_static_facts.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "nlm_grounded_static_facts",
        str(MODULE_PATH),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeAtom:
    def __init__(self, predicate, *args):
        self.predicate = predicate
        self.args = args

    def __hash__(self):
        return hash((self.predicate, self.args))

    def __eq__(self, other):
        return (
            isinstance(other, FakeAtom)
            and self.predicate == other.predicate
            and self.args == other.args
        )


class FakeAction:
    def __init__(self, deleted=()):
        self.del_effects = [((), fact) for fact in deleted]


class GroundedStaticFactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_preserves_only_groundings_that_cannot_be_deleted(self):
        fixed_locations = {
            FakeAtom("at", "hoist0", "depot0"),
            FakeAtom("at", "hoist1", "distributor0"),
            FakeAtom("at", "pallet0", "distributor0"),
            FakeAtom("at", "pallet1", "depot0"),
            FakeAtom("at", "pallet2", "depot0"),
            FakeAtom("at", "pallet3", "depot0"),
        }
        crate_at = FakeAtom("at", "crate1", "depot0")
        truck_at = FakeAtom("at", "truck0", "depot0")
        object_equality = FakeAtom("=", "crate1", "crate1")

        result = self.module.collect_grounded_static_init_facts(
            list(fixed_locations)
            + [crate_at, truck_at, object_equality],
            [
                FakeAction(deleted=[crate_at]),
                FakeAction(deleted=[truck_at]),
            ],
        )

        self.assertEqual(result, fixed_locations)

    def test_add_only_or_redundant_facts_remain_true(self):
        always_true = FakeAtom("connected", "depot0", "distributor0")

        result = self.module.collect_grounded_static_init_facts(
            [always_true],
            [FakeAction()],
        )

        self.assertEqual(result, {always_true})


if __name__ == "__main__":
    unittest.main()
