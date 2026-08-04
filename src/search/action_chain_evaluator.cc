#include "action_chain_evaluator.h"

#include "evaluation_context.h"
#include "evaluation_result.h"
#include "global_operator.h"
#include "globals.h"
#include "heuristic.h"
#include "state_registry.h"

#include <chrono>
#include <cctype>

using namespace std;

namespace {
struct HeuristicValue {
    bool finite;
    double value;

    HeuristicValue(bool finite_, double value_)
        : finite(finite_), value(value_) {
    }
};

double elapsed_seconds(const chrono::steady_clock::time_point &started_at) {
    return chrono::duration_cast<chrono::duration<double>>(
               chrono::steady_clock::now() - started_at)
        .count();
}

bool deadline_expired(
    const chrono::steady_clock::time_point &started_at, double max_seconds) {
    return max_seconds > 0 && elapsed_seconds(started_at) >= max_seconds;
}

HeuristicValue evaluate_h(
    const GlobalState &state, double g, Heuristic &heuristic) {
    EvaluationContext context(state, g, false, nullptr);
    const EvaluationResult &evaluation = context.get_result(&heuristic);
    if (evaluation.is_infinite())
        return HeuristicValue(false, 0.0);
    return HeuristicValue(true, evaluation.get_h_value());
}

}

TrajectoryStateInfo::TrajectoryStateInfo()
    : state_index(0),
      state_id(0),
      has_h(false),
      h(0.0) {
}

ActionChainEvaluationResult::ActionChainEvaluationResult()
    : status("ok"),
      generated_action_count(0),
      applied_action_count(0),
      has_invalid_action(false),
      invalid_action_index(0),
      path_cost(0.0),
      registered_state_count(0),
      scorer_seconds(0.0) {
}

ActionChainEvaluator::ActionChainEvaluator() {
    operator_by_name.reserve(g_operators.size());
    for (const GlobalOperator &op : g_operators)
        operator_by_name[normalize_operator_name(op.get_name())] = &op;
}

string ActionChainEvaluator::normalize_operator_name(const string &raw_name) {
    size_t begin = 0;
    size_t end = raw_name.size();
    while (begin < end &&
           isspace(static_cast<unsigned char>(raw_name[begin])))
        ++begin;
    while (end > begin &&
           isspace(static_cast<unsigned char>(raw_name[end - 1])))
        --end;
    if (end > begin + 1 && raw_name[begin] == '(' &&
        raw_name[end - 1] == ')') {
        ++begin;
        --end;
    }

    string normalized;
    bool pending_space = false;
    for (size_t index = begin; index < end; ++index) {
        unsigned char ch = static_cast<unsigned char>(raw_name[index]);
        if (isspace(ch)) {
            pending_space = !normalized.empty();
            continue;
        }
        if (pending_space) {
            normalized += ' ';
            pending_space = false;
        }
        normalized += static_cast<char>(tolower(ch));
    }
    return normalized;
}

ActionResolution ActionChainEvaluator::resolve_action(
    const GlobalState &state, const string &raw_action) const {
    string requested_name = normalize_operator_name(raw_action);
    auto operator_it = operator_by_name.find(requested_name);
    if (operator_it == operator_by_name.end()) {
        return ActionResolution(
            ActionResolutionStatus::UNKNOWN_ACTION, nullptr);
    }
    const GlobalOperator *op = operator_it->second;
    if (!op->is_applicable(state)) {
        return ActionResolution(
            ActionResolutionStatus::INAPPLICABLE_ACTION, op);
    }
    return ActionResolution(ActionResolutionStatus::FOUND, op);
}

GlobalState ActionChainEvaluator::apply_action(
    const GlobalState &state, const GlobalOperator &op) const {
    return g_state_registry->get_successor_state(state, op);
}

ActionChainEvaluationResult ActionChainEvaluator::evaluate(
    const GlobalState &initial_state, const vector<string> &actions,
    Heuristic &heuristic, double max_seconds) const {
    const chrono::steady_clock::time_point started_at =
        chrono::steady_clock::now();
    ActionChainEvaluationResult result;
    result.generated_action_count = actions.size();
    result.states.reserve(actions.size() + 1);

    HeuristicValue initial_h = evaluate_h(initial_state, 0.0, heuristic);
    TrajectoryStateInfo initial_info;
    initial_info.state_id = initial_state.get_id().hash();
    initial_info.has_h = initial_h.finite;
    initial_info.h = initial_h.value;
    result.states.push_back(initial_info);

    auto finish = [&]() {
        result.registered_state_count = g_state_registry->size();
        result.scorer_seconds = elapsed_seconds(started_at);
        return result;
    };

    if (test_goal(initial_state)) {
        result.outcome = "goal_reached";
        return finish();
    }
    if (deadline_expired(started_at, max_seconds)) {
        result.status = "scorer_timeout";
        result.error_message = "trajectory scoring exceeded max_seconds";
        return finish();
    }

    GlobalState current_state = initial_state;
    double current_g = 0.0;
    for (size_t action_index = 0; action_index < actions.size();
         ++action_index) {
        if (deadline_expired(started_at, max_seconds)) {
            result.status = "scorer_timeout";
            result.error_message = "trajectory scoring exceeded max_seconds";
            return finish();
        }

        const string &raw_action = actions[action_index];
        ActionResolution resolution = resolve_action(current_state, raw_action);
        if (resolution.status != ActionResolutionStatus::FOUND) {
            result.outcome = "invalid";
            result.has_invalid_action = true;
            result.invalid_action_index = action_index;
            return finish();
        }

        const GlobalOperator &op = *resolution.op;
        double candidate_g = current_g + op.get_cost();
        GlobalState successor = apply_action(current_state, op);
        heuristic.reach_state(current_state, op, successor);
        HeuristicValue successor_h = evaluate_h(successor, candidate_g, heuristic);

        size_t state_index = result.states.size();
        TrajectoryStateInfo state_info;
        state_info.state_index = state_index;
        state_info.state_id = successor.get_id().hash();
        state_info.has_h = successor_h.finite;
        state_info.h = successor_h.value;
        result.states.push_back(state_info);

        ++result.applied_action_count;
        current_g = candidate_g;
        result.path_cost = current_g;
        current_state = successor;

        if (test_goal(successor)) {
            // A completion may contain redundant text after a valid plan.
            // Once the goal is reached, later actions are intentionally ignored.
            result.outcome = "goal_reached";
            return finish();
        }
        if (deadline_expired(started_at, max_seconds)) {
            result.status = "scorer_timeout";
            result.error_message = "trajectory scoring exceeded max_seconds";
            return finish();
        }
    }

    result.outcome = "legal_incomplete";
    return finish();
}
