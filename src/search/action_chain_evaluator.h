#ifndef ACTION_CHAIN_EVALUATOR_H
#define ACTION_CHAIN_EVALUATOR_H

#include "global_state.h"

#include <cstddef>
#include <string>
#include <unordered_map>
#include <vector>

class GlobalOperator;
class Heuristic;

enum class ActionResolutionStatus {
    FOUND,
    UNKNOWN_ACTION,
    INAPPLICABLE_ACTION
};

struct ActionResolution {
    ActionResolutionStatus status;
    const GlobalOperator *op;

    ActionResolution(ActionResolutionStatus status_, const GlobalOperator *op_)
        : status(status_), op(op_) {
    }
};

struct TrajectoryStateInfo {
    std::size_t state_index;
    std::size_t state_id;
    bool has_h;
    double h;

    TrajectoryStateInfo();
};

struct ActionChainEvaluationResult {
    std::string status;
    std::string outcome;
    std::string error_message;
    std::size_t generated_action_count;
    std::size_t applied_action_count;
    bool has_invalid_action;
    std::size_t invalid_action_index;
    double path_cost;
    std::size_t registered_state_count;
    double scorer_seconds;
    std::vector<TrajectoryStateInfo> states;

    ActionChainEvaluationResult();
};

/*
  Shared, search-independent action-chain simulator.

  Resolving and applying an action has no SearchNode/Open List side effects.
  The search injection path uses the two step-level methods below; the reward
  scorer uses evaluate() to collect trajectory-local state and heuristic data.
*/
class ActionChainEvaluator {
    std::unordered_map<std::string, const GlobalOperator *> operator_by_name;

public:
    ActionChainEvaluator();

    static std::string normalize_operator_name(const std::string &raw_name);

    ActionResolution resolve_action(
        const GlobalState &state, const std::string &raw_action) const;

    GlobalState apply_action(
        const GlobalState &state, const GlobalOperator &op) const;

    ActionChainEvaluationResult evaluate(
        const GlobalState &initial_state,
        const std::vector<std::string> &actions,
        Heuristic &heuristic,
        double max_seconds) const;
};

#endif
