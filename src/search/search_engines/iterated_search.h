#ifndef SEARCH_ENGINES_ITERATED_SEARCH_H
#define SEARCH_ENGINES_ITERATED_SEARCH_H

#include "../option_parser_util.h"
#include "../search_engine.h"

#include <chrono>
#include <memory>

namespace options {
class Options;
}

namespace iterated_search {
class IteratedSearch : public SearchEngine {
    const std::vector<ParseTree> engine_configs;
    bool pass_bound;
    bool repeat_last_phase;
    bool continue_on_fail;
    bool continue_on_solve;
    int phase;
    bool last_phase_found_solution;
    ap_float best_bound;
    bool iterated_found_solution;
    int incumbent_count;
    std::chrono::steady_clock::time_point anytime_started_at;
    bool has_anytime_deadline;
    std::chrono::steady_clock::time_point anytime_deadline;

    std::unique_ptr<SearchEngine> current_search;
    std::string current_search_name;


    std::unique_ptr<SearchEngine> get_search_engine(
        int engine_config_start_index);
    std::unique_ptr<SearchEngine> create_phase(int p);
    SearchStatus step_return_value();
    double elapsed_wall_seconds() const;
    double remaining_wall_seconds() const;

    virtual void initialize() override;
    virtual SearchStatus step() override;
public:
    explicit IteratedSearch(const Options &opts);
    virtual ~IteratedSearch() override = default;
    virtual void save_plan_if_necessary() const override;
    virtual void print_statistics() const override;
};
}

#endif
