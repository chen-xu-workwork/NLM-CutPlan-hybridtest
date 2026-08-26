#ifndef SEARCH_ENGINES_EAGER_SEARCH_H
#define SEARCH_ENGINES_EAGER_SEARCH_H

#include "../search_engine.h"

#include "../open_lists/open_list.h"

#include <memory>
#include <string>
#include <vector>

class ActionChainEvaluator;
class GlobalOperator;
class Heuristic;
class PruningMethod;
class ScalarEvaluator;

namespace options {
class Options;
}

namespace eager_search {
class LLMTriggerMonitor;

class EagerSearch : public SearchEngine {
    const bool reopen_closed_nodes;
    const bool use_multi_path_dependence;
    const int llm_h_open_list_key_index;
    Heuristic *llm_h_evaluator;

    std::unique_ptr<StateOpenList> open_list;
    ScalarEvaluator *f_evaluator;
    std::vector<ap_float> removed_key_buffer;

    std::vector<Heuristic *> heuristics;
    std::vector<Heuristic *> preferred_operator_heuristics;

    std::shared_ptr<PruningMethod> pruning_method;
    std::unique_ptr<LLMTriggerMonitor> llm_trigger_monitor;
    std::unique_ptr<ActionChainEvaluator> llm_action_chain_evaluator;

    std::pair<SearchNode, bool> fetch_next_node();
    void poll_llm_responses();
    bool inject_llm_action_chain(
        const std::string &request_id, StateID source_id,
        const std::vector<std::string> &actions);
    void requeue_llm_source(StateID source_id);
    void start_f_value_statistics(EvaluationContext &eval_context);
    void update_f_value_statistics(const SearchNode &node);
    void reward_progress();
    void print_checkpoint_line(int g) const;
    bool llm_ancestor_stagnant(
        const SearchNode &node, ap_float current_h);

protected:
    virtual void initialize() override;
    virtual SearchStatus step() override;

public:
    explicit EagerSearch(const options::Options &opts);
    virtual ~EagerSearch();

    virtual void set_anytime_iteration(int iteration) override;
    virtual void print_statistics() const override;

    void dump_search_space() const;
};
}

#endif
