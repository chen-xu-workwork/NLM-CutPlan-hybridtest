"""Identify initially true grounded facts that cannot become false."""

from __future__ import print_function


def collect_grounded_static_init_facts(init_facts, instantiated_actions):
    """Return source init atoms that no reachable grounded action can delete.

    Fast Downward classifies fluent predicates by predicate name. That is too
    coarse for typed domains where only some groundings are mutable. For
    example, ``at(crate0, depot0)`` can change in depots, while
    ``at(hoist0, depot0)`` cannot.

    The input contains only initially true propositional facts. Such a fact
    remains true for the whole search if no reachable grounded action has an
    exact delete effect for it. Add effects do not make an already true fact
    false, so they do not affect this test.

    Parser-generated object equalities are internal reasoning facts rather
    than user PDDL init predicates and are always excluded.
    """
    deletable_facts = set()
    for action in instantiated_actions:
        for _condition, fact in action.del_effects:
            deletable_facts.add(fact)

    return set(
        fact
        for fact in init_facts
        if getattr(fact, "predicate", None) != "="
        and fact not in deletable_facts
    )
