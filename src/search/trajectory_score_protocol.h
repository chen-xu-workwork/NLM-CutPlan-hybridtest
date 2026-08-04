#ifndef TRAJECTORY_SCORE_PROTOCOL_H
#define TRAJECTORY_SCORE_PROTOCOL_H

#include "action_chain_evaluator.h"

#include <cstddef>
#include <string>
#include <vector>

namespace trajectory_score_protocol {

const int PROTOCOL_VERSION = 1;

struct ScoreRequest {
    int protocol_version;
    std::string request_id;
    std::string problem_id;
    std::string task_hash;
    std::vector<std::string> actions;

    ScoreRequest();
};

enum class MessageKind {
    SCORE_REQUEST,
    SHUTDOWN,
    INVALID
};

struct ParsedMessage {
    MessageKind kind;
    ScoreRequest request;
    std::string error;

    ParsedMessage();
};

ParsedMessage parse_message(const std::string &json_line);

std::string serialize_ready(
    const std::string &task_hash,
    const std::string &heuristic_config,
    std::size_t max_request_bytes);

std::string serialize_score_response(
    const ScoreRequest &request,
    const std::string &loaded_task_hash,
    const ActionChainEvaluationResult &result,
    bool recycle_recommended,
    const std::string &recycle_reason);

std::string serialize_error_response(
    const ScoreRequest &request,
    const std::string &loaded_task_hash,
    const std::string &status,
    const std::string &error_message);

std::string serialize_shutdown_ack(
    const std::string &request_id,
    const std::string &task_hash);

}

#endif
