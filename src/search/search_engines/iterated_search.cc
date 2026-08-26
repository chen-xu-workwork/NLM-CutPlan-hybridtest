#include "iterated_search.h"

#include "../option_parser.h"
#include "../plugin.h"
#include "../utils/system.h"

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <memory>

using namespace std;

namespace iterated_search {
IteratedSearch::IteratedSearch(const Options &opts)
    : SearchEngine(opts),
      engine_configs(opts.get_list<ParseTree>("engine_configs")),
      pass_bound(opts.get<bool>("pass_bound")),
      repeat_last_phase(opts.get<bool>("repeat_last")),
      continue_on_fail(opts.get<bool>("continue_on_fail")),
      continue_on_solve(opts.get<bool>("continue_on_solve")),
      phase(0),
      last_phase_found_solution(false),
      best_bound(bound),
      iterated_found_solution(false),
      incumbent_count(0),
      has_anytime_deadline(false),
      current_search(nullptr) {
}

void IteratedSearch::initialize() {
    anytime_started_at = chrono::steady_clock::now();
    if (isfinite(max_time)) {
        has_anytime_deadline = true;
        anytime_deadline = anytime_started_at +
            chrono::duration_cast<chrono::steady_clock::duration>(
                chrono::duration<double>(max_time));
        // The outer SearchEngine cannot check its timer while step() is
        // blocked inside a child. Keep the deadline on both levels and pass
        // the same absolute point to every child below.
        set_wall_time_deadline(anytime_deadline);
    }
    const char *run_id = getenv("NLM_LLM_RUN_ID");
    cout << "[NLM-ANYTIME-RUN-START]"
         << " run_id=" << (run_id ? run_id : "standalone")
         << " pass_bound=" << (pass_bound ? 1 : 0)
         << " repeat_last=" << (repeat_last_phase ? 1 : 0)
         << " continue_on_solve=" << (continue_on_solve ? 1 : 0)
         << " continue_on_fail=" << (continue_on_fail ? 1 : 0)
         << " wall_time_limit_seconds="
         << (has_anytime_deadline ? max_time : -1)
         << endl;
}

double IteratedSearch::elapsed_wall_seconds() const {
    return chrono::duration_cast<chrono::duration<double>>(
        chrono::steady_clock::now() - anytime_started_at).count();
}

double IteratedSearch::remaining_wall_seconds() const {
    if (!has_anytime_deadline)
        return numeric_limits<double>::infinity();
    return max(0.0, chrono::duration_cast<chrono::duration<double>>(
        anytime_deadline - chrono::steady_clock::now()).count());
}

unique_ptr<SearchEngine> IteratedSearch::get_search_engine(
    int engine_configs_index) {
    OptionParser parser(engine_configs[engine_configs_index], false);
    unique_ptr<SearchEngine> engine(
        parser.start_parsing<SearchEngine *>());

    cout << "Starting search: ";
    kptree::print_tree_bracketed(engine_configs[engine_configs_index], cout);
    cout << endl;

    return engine;
}

unique_ptr<SearchEngine> IteratedSearch::create_phase(int p) {
    int num_phases = engine_configs.size();
    if (p >= num_phases) {
        /* We've gone through all searches. We continue if
           repeat_last_phase is true, but *not* if we didn't find a
           solution the last time around, since then this search would
           just behave the same way again (assuming determinism, which
           we might not actually have right now, but strive for). So
           this overrides continue_on_fail.
        */
        if (repeat_last_phase && last_phase_found_solution) {
            return get_search_engine(engine_configs.size() - 1);
        } else {
            return nullptr;
        }
    }

    return get_search_engine(p);
}

SearchStatus IteratedSearch::step() {
    current_search = create_phase(phase);
    if (!current_search) {
        return found_solution() ? SOLVED : FAILED;
    }
    if (has_anytime_deadline && remaining_wall_seconds() <= 0) {
        cout << "[NLM-ANYTIME-RUN-TIMEOUT]"
             << " elapsed_seconds=" << elapsed_wall_seconds()
             << " completed_iterations=" << phase
             << " incumbent_count=" << incumbent_count
             << " best_bound=" << best_bound
             << endl;
        current_search.reset();
        return TIMEOUT;
    }
    if (pass_bound) {
        current_search->set_bound(best_bound);
    }
    const int iteration = phase + 1;
    current_search->set_anytime_iteration(iteration);
    if (has_anytime_deadline)
        current_search->set_wall_time_deadline(anytime_deadline);
    cout << "[NLM-ANYTIME-PHASE-START]"
         << " iteration=" << iteration
         << " bound=" << best_bound
         << " remaining_seconds="
         << (has_anytime_deadline ? remaining_wall_seconds() : -1)
         << endl;
    ++phase;

    chrono::steady_clock::time_point phase_started_at =
        chrono::steady_clock::now();
    current_search->search();
    chrono::steady_clock::time_point solution_observed_at =
        chrono::steady_clock::now();
    double phase_seconds =
        chrono::duration_cast<chrono::duration<double>>(
            solution_observed_at - phase_started_at).count();
    double elapsed_seconds = elapsed_wall_seconds();

    SearchEngine::Plan found_plan;
    ap_float plan_cost = 0;
    last_phase_found_solution = current_search->found_solution();
    if (last_phase_found_solution) {
        iterated_found_solution = true;
        found_plan = current_search->get_plan();
        plan_cost = calculate_plan_cost(found_plan);
    }

    const SearchStatistics &current_stats = current_search->get_statistics();
    bool phase_timed_out = current_search->get_status() == TIMEOUT;
    const char *phase_result = last_phase_found_solution
        ? "solved"
        : (phase_timed_out ? "timeout" : "failed");
    cout << "[NLM-ANYTIME-PHASE-END]"
         << " iteration=" << iteration
         << " result=" << phase_result
         << " elapsed_seconds=" << elapsed_seconds
         << " phase_seconds=" << phase_seconds
         << " plan_cost="
         << (last_phase_found_solution ? plan_cost : -1)
         << " plan_length="
         << (last_phase_found_solution ? found_plan.size() : 0)
         << " phase_expanded=" << current_stats.get_expanded()
         << " phase_evaluated=" << current_stats.get_evaluated_states()
         << " phase_generated=" << current_stats.get_generated()
         << " phase_reopened=" << current_stats.get_reopened()
         << " peak_memory_kb=" << utils::get_peak_memory_in_kb()
         << endl;

    statistics.inc_expanded(current_stats.get_expanded());
    statistics.inc_evaluated_states(current_stats.get_evaluated_states());
    statistics.inc_evaluations(current_stats.get_evaluations());
    statistics.inc_generated(current_stats.get_generated());
    statistics.inc_generated_ops(current_stats.get_generated_ops());
    statistics.inc_reopened(current_stats.get_reopened());

    if (last_phase_found_solution && plan_cost < best_bound) {
        save_plan(found_plan, true);
        best_bound = plan_cost;
        set_plan(found_plan);
        ++incumbent_count;
        cout << "[NLM-ANYTIME-INCUMBENT]"
             << " iteration=" << iteration
             << " incumbent=" << incumbent_count
             << " elapsed_seconds=" << elapsed_seconds
             << " plan_cost=" << plan_cost
             << " plan_length=" << found_plan.size()
             << " plan_number=" << g_num_previously_generated_plans
             << " cumulative_expanded=" << statistics.get_expanded()
             << " cumulative_evaluated="
             << statistics.get_evaluated_states()
             << " cumulative_generated=" << statistics.get_generated()
             << " cumulative_reopened=" << statistics.get_reopened()
             << endl;
    }

    // Save/announce an incumbent before bridge cleanup so a process limit
    // cannot lose an already discovered plan. The phase-end event above has
    // already invalidated its Python futures; printing child statistics now
    // closes the old C++ sockets and reports final per-phase LLM counters.
    current_search->print_statistics();

    bool global_time_exhausted = has_anytime_deadline &&
        remaining_wall_seconds() <= 0;
    if (phase_timed_out && global_time_exhausted) {
        cout << "[NLM-ANYTIME-RUN-TIMEOUT]"
             << " elapsed_seconds=" << elapsed_wall_seconds()
             << " completed_iterations=" << phase
             << " incumbent_count=" << incumbent_count
             << " best_bound=" << best_bound
             << endl;
    }

    // A repeated-last anytime run can contain many phases. Releasing the
    // complete per-phase Open List/SearchSpace here prevents all previous
    // search trees and stopped bridge objects from accumulating in memory.
    current_search.reset();

    if (phase_timed_out && global_time_exhausted)
        return TIMEOUT;

    return step_return_value();
}

SearchStatus IteratedSearch::step_return_value() {
    if (iterated_found_solution)
        cout << "Best solution cost so far: " << best_bound << endl;

    if (last_phase_found_solution) {
        if (continue_on_solve) {
            cout << "Solution found - keep searching" << endl;
            return IN_PROGRESS;
        } else {
            cout << "Solution found - stop searching" << endl;
            return SOLVED;
        }
    } else {
        if (continue_on_fail) {
            cout << "No solution found - keep searching" << endl;
            return IN_PROGRESS;
        } else {
            cout << "No solution found - stop searching" << endl;
            return iterated_found_solution ? SOLVED : FAILED;
        }
    }
}

void IteratedSearch::print_statistics() const {
    cout << "Cumulative statistics:" << endl;
    statistics.print_detailed_statistics();
}

void IteratedSearch::save_plan_if_necessary() const {
    // We don't need to save here, as we automatically save after
    // each successful search iteration.
}

static SearchEngine *_parse(OptionParser &parser) {
    parser.document_synopsis("Iterated search", "");
    parser.document_note(
        "Note 1",
        "We do no cache values between search iterations at the moment. "
        "If you perform a LAMA-style iterative search, heuristic values "
        "will be computed multiple times. "
        "Adding heuristic caching is [issue108 http://issues.fast-downward.org/issue108].");
    parser.document_note(
        "Note 2",
        "The configuration\n```\n"
        "--search \"iterated([lazy_wastar(merge_and_shrink(),w=10), "
        "lazy_wastar(merge_and_shrink(),w=5), lazy_wastar(merge_and_shrink(),w=3), "
        "lazy_wastar(merge_and_shrink(),w=2), lazy_wastar(merge_and_shrink(),w=1)])\"\n"
        "```\nwould perform the preprocessing phase of the merge and shrink heuristic "
        "5 times (once before each iteration).\n\n"
        "To avoid this, use heuristic predefinition, which avoids duplicate "
        "preprocessing, as follows:\n```\n"
        "--heuristic \"h=merge_and_shrink()\" --search "
        "\"iterated([lazy_wastar(h,w=10), lazy_wastar(h,w=5), lazy_wastar(h,w=3), "
        "lazy_wastar(h,w=2), lazy_wastar(h,w=1)])\"\n"
        "```");
    parser.document_note(
        "Note 3",
        "If you reuse the same landmark count heuristic "
        "(using heuristic predefinition) between iterations, "
        "the path data (that is, landmark status for each visited state) "
        "will be saved between iterations.");
    parser.add_list_option<ParseTree>("engine_configs",
                                      "list of search engines for each phase");
    parser.add_option<bool>(
        "pass_bound",
        "use bound from previous search. The bound is the real cost "
        "of the plan found before, regardless of the cost_type parameter.",
        "true");
    parser.add_option<bool>("repeat_last",
                            "repeat last phase of search",
                            "false");
    parser.add_option<bool>("continue_on_fail",
                            "continue search after no solution found",
                            "false");
    parser.add_option<bool>("continue_on_solve",
                            "continue search after solution found",
                            "true");
    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();

    opts.verify_list_non_empty<ParseTree>("engine_configs");

    if (parser.help_mode()) {
        return nullptr;
    } else if (parser.dry_run()) {
        //check if the supplied search engines can be parsed
        for (const ParseTree &config : opts.get_list<ParseTree>("engine_configs")) {
            OptionParser test_parser(config, true);
            test_parser.start_parsing<SearchEngine *>();
        }
        return nullptr;
    } else {
        return new IteratedSearch(opts);
    }
}

static Plugin<SearchEngine> _plugin("iterated", _parse);
}
