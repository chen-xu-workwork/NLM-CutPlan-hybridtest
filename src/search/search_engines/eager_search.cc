#include "eager_search.h"

#include "search_common.h"

#include "../action_chain_evaluator.h"
#include "../evaluation_context.h"
#include "../globals.h"
#include "../heuristic.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../pruning_method.h"
#include "../successor_generator.h"
#include "../utils/system.h"
#include "../utils/timer.h"
#include "../utils/planvis.h"

#include "../open_lists/open_list_factory.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <cctype>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <mutex>
#include <queue>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if !defined(_WIN32)
#include <netdb.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>
#endif

using namespace std;

namespace eager_search {
static bool env_enabled(const char *name, bool default_value = false) {
    const char *value = getenv(name);
    if (!value)
        return default_value;

    string setting(value);
    return !setting.empty() && setting != "0" && setting != "false" &&
           setting != "FALSE";
}

static int env_int(const char *name, int default_value) {
    const char *value = getenv(name);
    if (!value)
        return default_value;
    return atoi(value);
}

static ap_float env_float(const char *name, ap_float default_value) {
    const char *value = getenv(name);
    if (!value)
        return default_value;
    return atof(value);
}

static string env_string(const char *name, const string &default_value) {
    const char *value = getenv(name);
    if (!value)
        return default_value;
    return string(value);
}

static bool env_equals_ignore_case(const char *name, const string &expected) {
    string value = env_string(name, "");
    transform(value.begin(), value.end(), value.begin(),
              [](unsigned char ch) { return static_cast<char>(tolower(ch)); });
    return value == expected;
}

static string state_id_label(StateID state_id) {
    ostringstream stream;
    stream << state_id;
    return stream.str();
}

static string json_escape(const string &value) {
    ostringstream out;
    for (char ch : value) {
        switch (ch) {
        case '"':
            out << "\\\"";
            break;
        case '\\':
            out << "\\\\";
            break;
        case '\b':
            out << "\\b";
            break;
        case '\f':
            out << "\\f";
            break;
        case '\n':
            out << "\\n";
            break;
        case '\r':
            out << "\\r";
            break;
        case '\t':
            out << "\\t";
            break;
        default:
            if (static_cast<unsigned char>(ch) < 0x20) {
                out << "\\u";
                const char *hex = "0123456789abcdef";
                out << "00"
                    << hex[(static_cast<unsigned char>(ch) >> 4) & 0x0f]
                    << hex[static_cast<unsigned char>(ch) & 0x0f];
            } else {
                out << ch;
            }
        }
    }
    return out.str();
}

struct LLMRequest {
    string request_id;
    string run_id;
    int iteration;
    StateID state_id;
    size_t state_index;
    string state_label;
    string problem_id;
    string reason;
    ap_float g;
    ap_float h;
    int search_expansions;
    string init;

    LLMRequest()
        : iteration(1), state_id(StateID::no_state), state_index(0), g(0), h(0),
          search_expansions(0) {
    }
};

struct LLMResponse {
    string request_id;
    StateID state_id;
    size_t state_index;
    string state_label;
    bool transport_ok;
    int http_status;
    string body;
    string error;

    LLMResponse()
        : state_id(StateID::no_state),
          state_index(0),
          transport_ok(false),
          http_status(0) {
    }
};

struct ParsedLLMResponse {
    bool valid;
    string status;
    vector<string> actions;
    vector<vector<string>> action_chains;
    string error;

    ParsedLLMResponse()
        : valid(false) {
    }
};

static void skip_json_whitespace(const string &text, size_t &position) {
    while (position < text.size() &&
           isspace(static_cast<unsigned char>(text[position]))) {
        ++position;
    }
}

static bool parse_json_string(const string &text, size_t &position,
                              string &result, string &error) {
    skip_json_whitespace(text, position);
    if (position >= text.size() || text[position] != '"') {
        error = "expected JSON string";
        return false;
    }
    ++position;
    result.clear();
    while (position < text.size()) {
        char ch = text[position++];
        if (ch == '"')
            return true;
        if (ch != '\\') {
            result += ch;
            continue;
        }
        if (position >= text.size()) {
            error = "truncated JSON escape";
            return false;
        }
        char escaped = text[position++];
        switch (escaped) {
        case '"': result += '"'; break;
        case '\\': result += '\\'; break;
        case '/': result += '/'; break;
        case 'b': result += '\b'; break;
        case 'f': result += '\f'; break;
        case 'n': result += '\n'; break;
        case 'r': result += '\r'; break;
        case 't': result += '\t'; break;
        case 'u':
            if (position + 4 > text.size()) {
                error = "truncated JSON unicode escape";
                return false;
            }
            // Operator names and object identifiers are ASCII. Preserve the
            // parser position for other response fields without introducing
            // a full Unicode dependency into the search binary.
            position += 4;
            result += '?';
            break;
        default:
            error = "invalid JSON escape";
            return false;
        }
    }
    error = "unterminated JSON string";
    return false;
}

static bool skip_json_value(const string &text, size_t &position, string &error);

static bool skip_json_array(const string &text, size_t &position, string &error) {
    ++position;
    skip_json_whitespace(text, position);
    if (position < text.size() && text[position] == ']') {
        ++position;
        return true;
    }
    while (position < text.size()) {
        if (!skip_json_value(text, position, error))
            return false;
        skip_json_whitespace(text, position);
        if (position < text.size() && text[position] == ',') {
            ++position;
            continue;
        }
        if (position < text.size() && text[position] == ']') {
            ++position;
            return true;
        }
        error = "expected ',' or ']' in JSON array";
        return false;
    }
    error = "unterminated JSON array";
    return false;
}

static bool skip_json_object(const string &text, size_t &position, string &error) {
    ++position;
    skip_json_whitespace(text, position);
    if (position < text.size() && text[position] == '}') {
        ++position;
        return true;
    }
    while (position < text.size()) {
        string ignored_key;
        if (!parse_json_string(text, position, ignored_key, error))
            return false;
        skip_json_whitespace(text, position);
        if (position >= text.size() || text[position] != ':') {
            error = "expected ':' in JSON object";
            return false;
        }
        ++position;
        if (!skip_json_value(text, position, error))
            return false;
        skip_json_whitespace(text, position);
        if (position < text.size() && text[position] == ',') {
            ++position;
            continue;
        }
        if (position < text.size() && text[position] == '}') {
            ++position;
            return true;
        }
        error = "expected ',' or '}' in JSON object";
        return false;
    }
    error = "unterminated JSON object";
    return false;
}

static bool skip_json_value(const string &text, size_t &position, string &error) {
    skip_json_whitespace(text, position);
    if (position >= text.size()) {
        error = "missing JSON value";
        return false;
    }
    if (text[position] == '"') {
        string ignored;
        return parse_json_string(text, position, ignored, error);
    }
    if (text[position] == '[')
        return skip_json_array(text, position, error);
    if (text[position] == '{')
        return skip_json_object(text, position, error);

    size_t start = position;
    while (position < text.size() &&
           text[position] != ',' && text[position] != ']' &&
           text[position] != '}' &&
           !isspace(static_cast<unsigned char>(text[position]))) {
        ++position;
    }
    if (position == start) {
        error = "invalid JSON value";
        return false;
    }
    return true;
}

static bool parse_json_string_array(const string &text, size_t &position,
                                    vector<string> &result, string &error) {
    skip_json_whitespace(text, position);
    if (position >= text.size() || text[position] != '[') {
        error = "actions must be a JSON array";
        return false;
    }
    ++position;
    skip_json_whitespace(text, position);
    if (position < text.size() && text[position] == ']') {
        ++position;
        return true;
    }
    while (position < text.size()) {
        string item;
        if (!parse_json_string(text, position, item, error))
            return false;
        result.push_back(item);
        skip_json_whitespace(text, position);
        if (position < text.size() && text[position] == ',') {
            ++position;
            continue;
        }
        if (position < text.size() && text[position] == ']') {
            ++position;
            return true;
        }
        error = "expected ',' or ']' in actions array";
        return false;
    }
    error = "unterminated actions array";
    return false;
}

static bool parse_json_string_matrix(
    const string &text, size_t &position,
    vector<vector<string>> &result, string &error) {
    skip_json_whitespace(text, position);
    if (position >= text.size() || text[position] != '[') {
        error = "action_chains must be a JSON array";
        return false;
    }
    ++position;
    skip_json_whitespace(text, position);
    if (position < text.size() && text[position] == ']') {
        ++position;
        return true;
    }
    while (position < text.size()) {
        vector<string> chain;
        if (!parse_json_string_array(text, position, chain, error))
            return false;
        result.push_back(chain);
        skip_json_whitespace(text, position);
        if (position < text.size() && text[position] == ',') {
            ++position;
            continue;
        }
        if (position < text.size() && text[position] == ']') {
            ++position;
            return true;
        }
        error = "expected ',' or ']' in action_chains array";
        return false;
    }
    error = "unterminated action_chains array";
    return false;
}

static ParsedLLMResponse parse_llm_response_body(const string &body) {
    ParsedLLMResponse result;
    size_t position = 0;
    skip_json_whitespace(body, position);
    if (position >= body.size() || body[position] != '{') {
        result.error = "response is not a JSON object";
        return result;
    }
    ++position;
    bool saw_status = false;
    bool saw_actions = false;
    bool saw_action_chains = false;
    bool object_closed = false;
    while (position < body.size()) {
        skip_json_whitespace(body, position);
        if (position < body.size() && body[position] == '}') {
            ++position;
            object_closed = true;
            break;
        }
        string key;
        if (!parse_json_string(body, position, key, result.error))
            return result;
        skip_json_whitespace(body, position);
        if (position >= body.size() || body[position] != ':') {
            result.error = "expected ':' after response field";
            return result;
        }
        ++position;
        if (key == "status") {
            if (!parse_json_string(body, position, result.status, result.error))
                return result;
            saw_status = true;
        } else if (key == "actions") {
            if (!parse_json_string_array(
                    body, position, result.actions, result.error)) {
                return result;
            }
            saw_actions = true;
        } else if (key == "action_chains") {
            if (!parse_json_string_matrix(
                    body, position, result.action_chains, result.error)) {
                return result;
            }
            saw_action_chains = true;
        } else if (!skip_json_value(body, position, result.error)) {
            return result;
        }
        skip_json_whitespace(body, position);
        if (position < body.size() && body[position] == ',') {
            ++position;
            continue;
        }
        if (position < body.size() && body[position] == '}') {
            ++position;
            object_closed = true;
            break;
        }
        result.error = "expected ',' or '}' in response object";
        return result;
    }
    if (!object_closed) {
        result.error = "unterminated response object";
        return result;
    }
    if (!saw_status || (!saw_actions && !saw_action_chains)) {
        result.error = "response is missing status and action payload";
        return result;
    }
    skip_json_whitespace(body, position);
    if (position != body.size()) {
        result.error = "response contains trailing content";
        return result;
    }
    if (!saw_action_chains)
        result.action_chains.push_back(result.actions);
    result.valid = true;
    return result;
}

class LLMBridge {
    // 中文说明：LLMBridge 是搜索线程和 Python 主控之间的通信边界。
    // 搜索线程只 submit/poll，HTTP 阻塞等待都发生在后台 worker 线程里。
    struct Config {
        bool enabled;
        string mode;
        string host;
        int port;
        string path;
        int timeout_ms;
        int worker_count;
        int max_queue;
        bool emit_state;

        Config()
            : enabled(env_enabled("NLM_LLM_TRIGGER")),
              mode(env_string("NLM_LLM_COMM_MODE", "log")),
              host(env_string("NLM_LLM_HTTP_HOST", "127.0.0.1")),
              port(env_int("NLM_LLM_HTTP_PORT", 8765)),
              path(env_string("NLM_LLM_HTTP_PATH", "/llm/request")),
              timeout_ms(max(1, env_int("NLM_LLM_HTTP_TIMEOUT_MS", 30000))),
              worker_count(max(1, env_int("NLM_LLM_HTTP_WORKERS", 8))),
              max_queue(env_int("NLM_LLM_HTTP_MAX_QUEUE", 0)),
              emit_state(env_enabled("NLM_LLM_EMIT_STATE")) {
            transform(mode.begin(), mode.end(), mode.begin(),
                      [](unsigned char ch) {
                          return static_cast<char>(tolower(ch));
                      });
        }
    };

    Config config;
    mutable mutex queue_mutex;
    condition_variable queue_cv;
    queue<LLMRequest> outgoing;
    deque<LLMResponse> completed;
    vector<thread> worker_threads;
    size_t active_requests;
    bool stopping;
    size_t queued_discarded_on_stop;
    size_t active_cancelled_on_stop;
    size_t completed_unconsumed_on_stop;
#if !defined(_WIN32)
    mutable mutex socket_mutex;
    unordered_set<int> active_sockets;
#endif

    bool http_mode() const {
        return config.enabled && config.mode == "http";
    }

    static string make_request_body(const LLMRequest &request) {
        ostringstream body;
        body << "{";
        body << "\"type\":\"llm_request\",";
        body << "\"request_id\":\"" << json_escape(request.request_id) << "\",";
        body << "\"run_id\":\"" << json_escape(request.run_id) << "\",";
        body << "\"iteration\":" << request.iteration << ",";
        body << "\"state_id\":" << request.state_index << ",";
        body << "\"state_label\":\"" << json_escape(request.state_label) << "\",";
        body << "\"problem_id\":\"" << json_escape(request.problem_id) << "\",";
        body << "\"reason\":\"" << json_escape(request.reason) << "\",";
        body << "\"g\":" << request.g << ",";
        body << "\"h\":" << request.h << ",";
        body << "\"search_expansions\":"
             << request.search_expansions << ",";
        body << "\"init\":\"" << json_escape(request.init) << "\"";
        body << "}";
        return body.str();
    }

    string make_http_request(const string &body) const {
        ostringstream request;
        request << "POST " << config.path << " HTTP/1.1\r\n";
        request << "Host: " << config.host << ":" << config.port << "\r\n";
        request << "Content-Type: application/json\r\n";
        request << "Accept: application/json\r\n";
        request << "Connection: close\r\n";
        request << "Content-Length: " << body.size() << "\r\n";
        request << "\r\n";
        request << body;
        return request.str();
    }

    static int parse_http_status(const string &response) {
        if (response.size() < 12 || response.compare(0, 5, "HTTP/") != 0)
            return 0;
        size_t first_space = response.find(' ');
        if (first_space == string::npos)
            return 0;
        return atoi(response.c_str() + first_space + 1);
    }

    static string parse_http_body(const string &response) {
        size_t split = response.find("\r\n\r\n");
        if (split == string::npos)
            return "";
        return response.substr(split + 4);
    }

    void push_completed(const LLMResponse &response) {
        lock_guard<mutex> lock(queue_mutex);
        assert(active_requests > 0);
        --active_requests;
        completed.push_back(response);
    }

#if !defined(_WIN32)
    bool register_socket_if_running(int fd) {
        lock_guard<mutex> queue_lock(queue_mutex);
        if (stopping)
            return false;
        lock_guard<mutex> socket_lock(socket_mutex);
        active_sockets.insert(fd);
        return true;
    }

    void close_registered_socket(int fd) {
        lock_guard<mutex> lock(socket_mutex);
        active_sockets.erase(fd);
        close(fd);
    }

    bool send_all(int fd, const string &data, string &error) const {
        size_t sent = 0;
        while (sent < data.size()) {
            int flags = 0;
#if defined(MSG_NOSIGNAL)
            flags = MSG_NOSIGNAL;
#endif
            ssize_t n = send(
                fd, data.data() + sent, data.size() - sent, flags);
            if (n <= 0) {
                error = string("send failed: ") + strerror(errno);
                return false;
            }
            sent += static_cast<size_t>(n);
        }
        return true;
    }

    LLMResponse post_request_posix(const LLMRequest &request) {
        LLMResponse response;
        response.request_id = request.request_id;
        response.state_id = request.state_id;
        response.state_index = request.state_index;
        response.state_label = request.state_label;

        string port_string = to_string(config.port);
        struct addrinfo hints;
        memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_UNSPEC;
        hints.ai_socktype = SOCK_STREAM;

        struct addrinfo *addresses = nullptr;
        int gai_result = getaddrinfo(
            config.host.c_str(), port_string.c_str(), &hints, &addresses);
        if (gai_result != 0) {
            response.error = string("getaddrinfo failed: ") +
                             gai_strerror(gai_result);
            return response;
        }

        int fd = -1;
        for (struct addrinfo *addr = addresses; addr; addr = addr->ai_next) {
            fd = socket(addr->ai_family, addr->ai_socktype, addr->ai_protocol);
            if (fd == -1)
                continue;
            if (!register_socket_if_running(fd)) {
                close(fd);
                fd = -1;
                response.error = "bridge stopping";
                break;
            }

            struct timeval timeout;
            timeout.tv_sec = config.timeout_ms / 1000;
            timeout.tv_usec = (config.timeout_ms % 1000) * 1000;
            setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
                       reinterpret_cast<const char *>(&timeout),
                       sizeof(timeout));
            setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
                       reinterpret_cast<const char *>(&timeout),
                       sizeof(timeout));

            if (connect(fd, addr->ai_addr, addr->ai_addrlen) == 0)
                break;

            close_registered_socket(fd);
            fd = -1;
        }
        freeaddrinfo(addresses);

        if (fd == -1) {
            if (response.error.empty())
                response.error = "connect failed";
            return response;
        }

        string body = make_request_body(request);
        string http_request = make_http_request(body);
        string send_error;
        if (!send_all(fd, http_request, send_error)) {
            response.error = send_error;
            close_registered_socket(fd);
            return response;
        }

        string raw_response;
        char buffer[4096];
        while (true) {
            ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
            if (n > 0) {
                raw_response.append(buffer, static_cast<size_t>(n));
            } else if (n == 0) {
                break;
            } else {
                response.error = string("recv failed: ") + strerror(errno);
                close_registered_socket(fd);
                return response;
            }
        }
        close_registered_socket(fd);

        response.http_status = parse_http_status(raw_response);
        response.body = parse_http_body(raw_response);
        response.transport_ok =
            response.http_status >= 200 && response.http_status < 300;
        if (!response.transport_ok && response.error.empty()) {
            ostringstream error;
            error << "http status " << response.http_status;
            response.error = error.str();
        }
        return response;
    }
#else
    LLMResponse post_request_windows_stub(const LLMRequest &request) const {
        LLMResponse response;
        response.request_id = request.request_id;
        response.state_id = request.state_id;
        response.state_index = request.state_index;
        response.state_label = request.state_label;
        response.error = "HTTP bridge is only implemented for POSIX/WSL builds";
        return response;
    }
#endif

    LLMResponse post_request(const LLMRequest &request) {
#if !defined(_WIN32)
        return post_request_posix(request);
#else
        return post_request_windows_stub(request);
#endif
    }

    void worker_loop(int worker_index) {
        while (true) {
            LLMRequest request;
            size_t in_flight = 0;
            size_t queued = 0;
            {
                unique_lock<mutex> lock(queue_mutex);
                queue_cv.wait(lock, [this]() {
                    return stopping || !outgoing.empty();
                });
                if (stopping)
                    return;
                request = outgoing.front();
                outgoing.pop();
                ++active_requests;
                in_flight = active_requests;
                queued = outgoing.size();
            }

            ostringstream dispatch_log;
            dispatch_log << "[NLM-LLM-BRIDGE] dispatched"
                         << " worker=" << worker_index
                         << " request_id=" << request.request_id
                         << " state=" << request.state_label
                         << " in_flight=" << in_flight
                         << " queued=" << queued;
            cout << dispatch_log.str() << endl;

            LLMResponse response = post_request(request);
            push_completed(response);
        }
    }

    void log_request(const LLMRequest &request) const {
        cout << "[NLM-LLM-TRIGGER] request"
             << " run_id=" << request.run_id
             << " iteration=" << request.iteration
             << " request_id=" << request.request_id
             << " state=" << request.state_label
             << " reason=" << request.reason
             << " g=" << request.g
             << " h=" << request.h
             << " expansions=" << request.search_expansions
             << " comm_mode=" << config.mode
             << endl;
        if (!config.emit_state)
            return;
        cout << "[NLM-LLM-TRIGGER-STATE] begin state="
             << request.state_label << endl;
        cout << request.init;
        cout << "[NLM-LLM-TRIGGER-STATE] end state="
             << request.state_label << endl;
    }

public:
    LLMBridge()
        : active_requests(0),
          stopping(false),
          queued_discarded_on_stop(0),
          active_cancelled_on_stop(0),
          completed_unconsumed_on_stop(0) {
    }

    ~LLMBridge() {
        stop();
    }

    void start() {
        if (!config.enabled)
            return;
        if (http_mode()) {
            cout << "[NLM-LLM-BRIDGE] starting http worker pool"
                 << " host=" << config.host
                 << " port=" << config.port
                 << " path=" << config.path
                 << " timeout_ms=" << config.timeout_ms
                 << " workers=" << config.worker_count
                 << " max_queue=" << config.max_queue
                 << endl;
            worker_threads.reserve(config.worker_count);
            for (int index = 0; index < config.worker_count; ++index)
                worker_threads.emplace_back(
                    &LLMBridge::worker_loop, this, index);
        } else {
            cout << "[NLM-LLM-BRIDGE] using log mode"
                 << " comm_mode=" << config.mode
                 << endl;
        }
    }

    void stop() {
        {
            lock_guard<mutex> lock(queue_mutex);
            if (stopping)
                return;
            stopping = true;
            queued_discarded_on_stop += outgoing.size();
            active_cancelled_on_stop += active_requests;
            completed_unconsumed_on_stop += completed.size();
            queue<LLMRequest> empty;
            outgoing.swap(empty);
        }
        queue_cv.notify_all();
#if !defined(_WIN32)
        {
            lock_guard<mutex> lock(socket_mutex);
            for (int fd : active_sockets)
                shutdown(fd, SHUT_RDWR);
        }
#endif
        for (thread &worker : worker_threads) {
            if (worker.joinable())
                worker.join();
        }
        if (!worker_threads.empty()) {
            cout << "[NLM-LLM-BRIDGE] http worker pool stopped"
                 << " workers=" << worker_threads.size()
                 << endl;
            worker_threads.clear();
        }
    }

    bool submit(const LLMRequest &request) {
        if (!config.enabled)
            return false;
        if (!http_mode()) {
            log_request(request);
            return true;
        }

        {
            lock_guard<mutex> lock(queue_mutex);
            if (stopping)
                return false;
            if (config.max_queue > 0 &&
                static_cast<int>(outgoing.size()) >= config.max_queue) {
                cout << "[NLM-LLM-BRIDGE] outgoing queue full"
                     << " request_id=" << request.request_id
                     << " state=" << request.state_label
                     << endl;
                return false;
            }
            outgoing.push(request);
        }
        queue_cv.notify_one();
        cout << "[NLM-LLM-BRIDGE] submitted"
             << " request_id=" << request.request_id
             << " state=" << request.state_label
             << " reason=" << request.reason
             << endl;
        return true;
    }

    bool expects_response() const {
        return http_mode();
    }

    vector<LLMResponse> poll_completed() {
        vector<LLMResponse> responses;
        lock_guard<mutex> lock(queue_mutex);
        while (!completed.empty()) {
            responses.push_back(completed.front());
            completed.pop_front();
        }
        return responses;
    }

    size_t queued_discarded() const {
        lock_guard<mutex> lock(queue_mutex);
        return queued_discarded_on_stop;
    }

    size_t active_cancelled() const {
        lock_guard<mutex> lock(queue_mutex);
        return active_cancelled_on_stop;
    }

    size_t completed_unconsumed() const {
        lock_guard<mutex> lock(queue_mutex);
        return completed_unconsumed_on_stop;
    }
};

class LLMTriggerMonitor {
    // 中文说明：这个 monitor 是 LLM 介入判定的轻量旁路层。
    // 它不改变 openlist 的内部结构，也不保存逐状态副本；热路径只更新
    // 固定容量 h 桶中的几个计数器。不同 h 可以任意交错，平台分析和
    // 父链检查均按展开数稀疏执行，运行期间不增长监视器主体内存。
    // HTTP 模式会把候选交给 Python 主控；log 模式只记录触发。
    enum {
        MAX_ACTIVITY_WINDOWS = 8,
        LAYER_TABLE_CAPACITY = 16
    };

    struct Config {
        bool enabled;
        bool skip_pending;
        bool emit_state;
        bool log_response_body;
        bool request_initial;
        bool enable_ancestor_stagnation;
        bool enable_frontier_plateau;
        bool enable_global_stall;
        int analysis_interval;
        int activity_windows;
        int growth_confirm_windows;
        int layer_reset_windows;
        int layer_min_recent_expanded;
        int layer_min_recent_net_growth;
        int layer_min_since_request_expanded;
        int layer_min_since_request_net_growth;
        int stall_expansions;
        int ancestor_check_interval;
        int ancestor_depth;
        int min_depth;
        int min_request_gap_expansions;
        int per_layer_request_gap_expansions;
        int candidate_layers;
        int requests_per_slot;
        int heartbeat_interval;
        int max_pending;
        int max_requests;
        ap_float h_abs_epsilon;
        ap_float h_relative_epsilon;
        ap_float plateau_growth_ratio;

        Config()
            : enabled(env_enabled("NLM_LLM_TRIGGER")),
              skip_pending(env_equals_ignore_case(
                  "NLM_LLM_PENDING_BEHAVIOR", "skip")),
              emit_state(env_enabled("NLM_LLM_EMIT_STATE")),
              log_response_body(env_enabled("NLM_LLM_LOG_RESPONSE_BODY")),
              request_initial(env_enabled("NLM_LLM_REQUEST_INITIAL")),
              enable_ancestor_stagnation(env_enabled(
                  "NLM_LLM_ENABLE_ANCESTOR_STAGNATION", true)),
              enable_frontier_plateau(env_enabled(
                  "NLM_LLM_ENABLE_FRONTIER_PLATEAU", true)),
              enable_global_stall(env_enabled(
                  "NLM_LLM_ENABLE_GLOBAL_STALL", true)),
              analysis_interval(max(
                  1, env_int("NLM_LLM_ANALYSIS_INTERVAL", 8192))),
              activity_windows(min<int>(
                  MAX_ACTIVITY_WINDOWS, max(
                      1, env_int("NLM_LLM_ACTIVITY_WINDOWS", 4)))),
              growth_confirm_windows(max(
                  1, env_int("NLM_LLM_GROWTH_CONFIRM_WINDOWS", 2))),
              layer_reset_windows(max(
                  1, env_int("NLM_LLM_LAYER_RESET_WINDOWS", 4))),
              layer_min_recent_expanded(max(
                  1, env_int(
                      "NLM_LLM_LAYER_MIN_RECENT_EXPANDED", 4096))),
              layer_min_recent_net_growth(max(
                  1, env_int(
                      "NLM_LLM_LAYER_MIN_RECENT_NET_GROWTH", 1024))),
              layer_min_since_request_expanded(max(
                  1, env_int(
                      "NLM_LLM_LAYER_MIN_SINCE_REQUEST_EXPANDED", 8192))),
              layer_min_since_request_net_growth(max(
                  1, env_int(
                      "NLM_LLM_LAYER_MIN_SINCE_REQUEST_NET_GROWTH", 2048))),
              stall_expansions(env_int(
                  "NLM_LLM_STALL_EXPANSIONS", 500000)),
              ancestor_check_interval(max(
                  1, env_int("NLM_LLM_ANCESTOR_CHECK_INTERVAL", 100000))),
              ancestor_depth(max(1, env_int("NLM_LLM_ANCESTOR_DEPTH", 10))),
              min_depth(max(0, env_int("NLM_LLM_MIN_DEPTH", 20))),
              min_request_gap_expansions(max(
                  0, env_int("NLM_LLM_MIN_REQUEST_GAP_EXPANSIONS", 100000))),
              per_layer_request_gap_expansions(max(
                  0, env_int(
                      "NLM_LLM_PER_LAYER_REQUEST_GAP_EXPANSIONS", 500000))),
              candidate_layers(min<int>(
                  LAYER_TABLE_CAPACITY, max(
                      1, env_int("NLM_LLM_CANDIDATE_LAYERS", 3)))),
              requests_per_slot(min<int>(
                  LAYER_TABLE_CAPACITY, max(
                      1, env_int("NLM_LLM_REQUESTS_PER_SLOT", 1)))),
              heartbeat_interval(max(
                  0, env_int("NLM_LLM_HEARTBEAT_INTERVAL", 100000))),
              max_pending(env_int("NLM_LLM_MAX_PENDING", 0)),
              max_requests(max(0, env_int("NLM_LLM_MAX_REQUESTS", 10))),
              h_abs_epsilon(max(
                  0.0, env_float("NLM_LLM_H_EPSILON", 0.001))),
              h_relative_epsilon(max(
                  0.0, env_float("NLM_LLM_H_RELATIVE_EPSILON", 0.005))),
              plateau_growth_ratio(max(
                  1.0, env_float("NLM_LLM_PLATEAU_GROWTH_RATIO", 1.05))) {
            per_layer_request_gap_expansions = max(
                per_layer_request_gap_expansions,
                min_request_gap_expansions);
            requests_per_slot = min(requests_per_slot, candidate_layers);
        }
    };

    struct PendingRequestInfo {
        chrono::steady_clock::time_point submitted_at;
        int expansions_at_submit;
        string reason;

        PendingRequestInfo()
            : submitted_at(chrono::steady_clock::now()),
              expansions_at_submit(0) {
        }

        PendingRequestInfo(int expansions_, const string &reason_)
            : submitted_at(chrono::steady_clock::now()),
              expansions_at_submit(expansions_), reason(reason_) {
        }
    };

    struct LayerStats {
        bool occupied;
        int64_t h_key;
        ap_float h;
        size_t current_opened;
        size_t current_expanded;
        array<size_t, MAX_ACTIVITY_WINDOWS> opened_history;
        array<size_t, MAX_ACTIVITY_WINDOWS> expanded_history;
        size_t recent_opened;
        size_t recent_expanded;
        size_t opened_since_request;
        size_t expanded_since_request;
        int last_seen_expansion;
        int last_request_expansion;
        int growth_streak;
        int calm_streak;
        int inactive_streak;
        int requests_in_episode;
        bool growing;
        bool ready;
        StateID representative_state;
        ap_float representative_g;
        int representative_expansion;

        LayerStats()
            : occupied(false),
              h_key(0),
              h(0),
              current_opened(0),
              current_expanded(0),
              recent_opened(0),
              recent_expanded(0),
              opened_since_request(0),
              expanded_since_request(0),
              last_seen_expansion(-1),
              last_request_expansion(-1),
              growth_streak(0),
              calm_streak(0),
              inactive_streak(0),
              requests_in_episode(0),
              growing(false),
              ready(false),
              representative_state(StateID::no_state),
              representative_g(0),
              representative_expansion(-1) {
            opened_history.fill(0);
            expanded_history.fill(0);
        }
    };

    Config config;
    unordered_set<StateID> pending_states;
    unordered_set<StateID> suspended_states;
    unordered_set<StateID> requested_states;
    unordered_map<StateID, PendingRequestInfo> pending_request_infos;
    LLMBridge bridge;
    string run_id;
    int anytime_iteration;
    int next_request_id;
    int expansions;
    int expansions_since_best_h;
    int last_analysis_expansion;
    int last_ancestor_check_expansion;
    int last_heartbeat_expansion;
    ap_float best_h;
    bool global_stall_condition_active;
    bool global_stall_requested;
    array<LayerStats, LAYER_TABLE_CAPACITY> layers;
    int history_cursor;
    size_t opened_states;
    size_t analysis_checks;
    size_t frontier_plateau_events;
    size_t layer_episode_resets;
    size_t layer_table_evictions;
    size_t layer_requests_submitted;
    size_t global_stall_events;
    size_t ancestor_checks;
    size_t ancestor_deferrals;
    size_t ancestor_stagnation_events;
    size_t request_attempts;
    size_t requests_submitted;
    size_t requests_rejected_duplicate;
    size_t requests_rejected_pending_limit;
    size_t requests_rejected_request_limit;
    size_t requests_rejected_spacing;
    size_t requests_rejected_bridge;
    size_t responses_completed;
    size_t response_transport_failures;
    size_t usable_responses;
    size_t injected_chains;
    size_t injected_actions;
    size_t injected_states;
    size_t responses_discarded_phase_end;
    size_t completed_unconsumed_phase_end;
    size_t max_pending_observed;
    double total_response_seconds;
    double max_response_seconds;
    size_t total_response_age_expansions;
    size_t max_response_age_expansions;
    int first_request_expansion;
    int last_request_expansion;
    bool statistics_printed;
    unordered_map<string, size_t> request_attempts_by_reason;
    unordered_map<string, size_t> requests_submitted_by_reason;
    unordered_map<string, size_t> responses_completed_by_reason;

    bool reached_pending_limit() const {
        return config.max_pending > 0 &&
               static_cast<int>(pending_states.size()) >= config.max_pending;
    }

    bool reached_request_limit() const {
        return config.max_requests > 0 &&
               static_cast<int>(requests_submitted) >= config.max_requests;
    }

    bool request_spacing_active() const {
        return last_request_expansion >= 0 &&
               expansions - last_request_expansion <
                   config.min_request_gap_expansions;
    }

    ap_float h_improvement_threshold(ap_float reference_h,
                                     ap_float current_h) const {
        // 中文说明：h 的量纲会随问题变化，所以停滞判断使用相对阈值，
        // 再用一个很小的绝对阈值兜住 h 接近 0 或浮点误差的情况。
        if (!std::isfinite(static_cast<double>(reference_h)) ||
            !std::isfinite(static_cast<double>(current_h))) {
            return config.h_abs_epsilon;
        }
        ap_float scale = max(std::fabs(reference_h), std::fabs(current_h));
        return max(config.h_abs_epsilon,
                   config.h_relative_epsilon * scale);
    }

    bool meaningfully_improves_h(ap_float reference_h,
                                 ap_float current_h) const {
        if (!std::isfinite(static_cast<double>(reference_h)))
            return true;
        ap_float improvement = reference_h - current_h;
        return improvement > h_improvement_threshold(reference_h, current_h);
    }

    LLMRequest make_request(StateID state_id, const string &reason,
                            ap_float g, ap_float h) {
        LLMRequest request;
        ostringstream request_id;
        request_id << run_id << "-p" << anytime_iteration << "-"
                   << state_id.hash() << "-" << next_request_id++;
        request.request_id = request_id.str();
        request.run_id = run_id;
        request.iteration = anytime_iteration;
        request.state_id = state_id;
        request.state_index = state_id.hash();
        request.state_label = state_id_label(state_id);
        request.problem_id = env_string(
            "NLM_LLM_PROBLEM_ID", env_string("NLM_PROBLEM_ID", ""));
        request.reason = reason;
        request.g = g;
        request.h = h;
        request.search_expansions = expansions;

        GlobalState state = g_state_registry->lookup_state(state_id);
        request.init = state.get_pddl_init_string();
        return request;
    }

    bool request_state(StateID state_id, const string &reason,
                       ap_float g, ap_float h,
                       bool bypass_spacing = false) {
        ++request_attempts;
        ++request_attempts_by_reason[reason];
        if (!config.enabled || state_id == StateID::no_state)
            return false;
        if (reached_request_limit()) {
            ++requests_rejected_request_limit;
            return false;
        }
        if (!bypass_spacing && request_spacing_active()) {
            ++requests_rejected_spacing;
            return false;
        }
        if (requested_states.count(state_id)) {
            ++requests_rejected_duplicate;
            return false;
        }
        if (reached_pending_limit()) {
            ++requests_rejected_pending_limit;
            return false;
        }

        LLMRequest request = make_request(state_id, reason, g, h);
        if (!bridge.submit(request)) {
            ++requests_rejected_bridge;
            return false;
        }

        ++requests_submitted;
        ++requests_submitted_by_reason[reason];
        if (first_request_expansion < 0)
            first_request_expansion = expansions;
        last_request_expansion = expansions;
        requested_states.insert(state_id);
        // Log mode is observational and never produces a completion. Treating
        // those requests as pending would make an empty Open List wait forever.
        if (bridge.expects_response()) {
            pending_states.insert(state_id);
            pending_request_infos[state_id] =
                PendingRequestInfo(expansions, reason);
            max_pending_observed =
                max(max_pending_observed, pending_states.size());
        }
        return true;
    }

    ap_float h_bucket_width() const {
        return max(config.h_abs_epsilon, static_cast<ap_float>(0.001));
    }

    int64_t quantize_h(ap_float h) const {
        return static_cast<int64_t>(llround(h / h_bucket_width()));
    }

    size_t hash_h_key(int64_t key) const {
        uint64_t value = static_cast<uint64_t>(key);
        value ^= value >> 33;
        value *= 0xff51afd7ed558ccdULL;
        value ^= value >> 33;
        return static_cast<size_t>(
            value & static_cast<uint64_t>(LAYER_TABLE_CAPACITY - 1));
    }

    void initialize_layer(LayerStats &layer, int64_t key, ap_float h) {
        layer = LayerStats();
        layer.occupied = true;
        layer.h_key = key;
        layer.h = h;
        layer.last_seen_expansion = expansions;
    }

    LayerStats &layer_for_h(ap_float h) {
        int64_t key = quantize_h(h);
        size_t start = hash_h_key(key);
        for (int probe = 0; probe < LAYER_TABLE_CAPACITY; ++probe) {
            size_t index = (start + probe) & (LAYER_TABLE_CAPACITY - 1);
            LayerStats &layer = layers[index];
            if (layer.occupied && layer.h_key == key)
                return layer;
            if (!layer.occupied) {
                initialize_layer(layer, key, h);
                return layer;
            }
        }

        // The table is deliberately fixed-size. Reaching this path means more
        // than 16 distinct h buckets have been active over the run; replace
        // the stalest bucket without allocating or growing memory.
        int victim = 0;
        for (int index = 1; index < LAYER_TABLE_CAPACITY; ++index) {
            if (layers[index].last_seen_expansion <
                layers[victim].last_seen_expansion) {
                victim = index;
            }
        }
        ++layer_table_evictions;
        initialize_layer(layers[victim], key, h);
        return layers[victim];
    }

    size_t layer_net_growth(const LayerStats &layer) const {
        return layer.recent_opened > layer.recent_expanded
            ? layer.recent_opened - layer.recent_expanded : 0;
    }

    double layer_growth_ratio(const LayerStats &layer) const {
        if (layer.recent_expanded == 0)
            return 0.0;
        return static_cast<double>(layer.recent_opened) /
               static_cast<double>(layer.recent_expanded);
    }

    size_t layer_since_request_net_growth(
        const LayerStats &layer) const {
        return layer.opened_since_request > layer.expanded_since_request
            ? layer.opened_since_request - layer.expanded_since_request : 0;
    }

    void reset_layer_episode(LayerStats &layer, bool reset_cooldown) {
        int previous_request_expansion = layer.last_request_expansion;
        layer.opened_history.fill(0);
        layer.expanded_history.fill(0);
        layer.recent_opened = 0;
        layer.recent_expanded = 0;
        layer.opened_since_request = 0;
        layer.expanded_since_request = 0;
        layer.last_request_expansion = reset_cooldown
            ? -1 : previous_request_expansion;
        layer.growth_streak = 0;
        layer.calm_streak = 0;
        layer.inactive_streak = 0;
        layer.requests_in_episode = 0;
        layer.growing = false;
        layer.ready = false;
        layer.representative_state = StateID::no_state;
        layer.representative_g = 0;
        layer.representative_expansion = -1;
        ++layer_episode_resets;
    }

    void update_layer_window(LayerStats &layer) {
        bool no_new_opened = layer.current_opened == 0;
        layer.opened_history[history_cursor] = layer.current_opened;
        layer.expanded_history[history_cursor] = layer.current_expanded;
        layer.current_opened = 0;
        layer.current_expanded = 0;
        if (no_new_opened)
            ++layer.inactive_streak;
        else
            layer.inactive_streak = 0;

        layer.recent_opened = 0;
        layer.recent_expanded = 0;
        for (int index = 0; index < config.activity_windows; ++index) {
            layer.recent_opened += layer.opened_history[index];
            layer.recent_expanded += layer.expanded_history[index];
        }

        size_t net_growth = layer_net_growth(layer);
        bool enough_samples =
            layer.recent_expanded >=
                static_cast<size_t>(config.layer_min_recent_expanded) &&
            net_growth >=
                static_cast<size_t>(config.layer_min_recent_net_growth);
        layer.growing = config.enable_frontier_plateau &&
            enough_samples &&
            layer_growth_ratio(layer) >= config.plateau_growth_ratio;

        if (layer.growing) {
            ++layer.growth_streak;
            layer.calm_streak = 0;
            if (!layer.ready &&
                layer.growth_streak >= config.growth_confirm_windows) {
                layer.ready = true;
                ++frontier_plateau_events;
            }
        } else {
            layer.growth_streak = 0;
            ++layer.calm_streak;
            bool has_episode_state =
                layer.ready || layer.requests_in_episode > 0 ||
                layer.opened_since_request > 0 ||
                layer.expanded_since_request > 0;
            bool inactive_long_enough =
                layer.inactive_streak >= config.layer_reset_windows;

            // An active-but-calm layer ends its episode while retaining the
            // longer per-layer cooldown. If that layer subsequently vanishes,
            // clear the retained cooldown as well so a genuine reappearance
            // starts a fresh episode. The second branch matters even when the
            // first reset already cleared the episode counters.
            if (inactive_long_enough &&
                (has_episode_state || layer.last_request_expansion >= 0)) {
                reset_layer_episode(layer, true);
            } else if (has_episode_state &&
                       layer.calm_streak >= config.layer_reset_windows) {
                reset_layer_episode(layer, false);
            }
        }
    }

    bool layer_has_new_evidence(const LayerStats &layer) const {
        if (layer.expanded_since_request <
            static_cast<size_t>(
                config.layer_min_since_request_expanded)) {
            return false;
        }
        return layer_since_request_net_growth(layer) >=
               static_cast<size_t>(
                   config.layer_min_since_request_net_growth);
    }

    bool layer_request_eligible(const LayerStats &layer) const {
        if (!config.enable_frontier_plateau || !layer.occupied ||
            !layer.ready || !layer.growing ||
            layer.representative_state == StateID::no_state ||
            requested_states.count(layer.representative_state) ||
            !layer_has_new_evidence(layer)) {
            return false;
        }
        if (layer.last_request_expansion >= 0 &&
            expansions - layer.last_request_expansion <
                config.per_layer_request_gap_expansions) {
            return false;
        }
        return layer.representative_expansion >
               layer.last_request_expansion;
    }

    bool has_eligible_layer_candidate() const {
        for (const LayerStats &layer : layers) {
            if (layer_request_eligible(layer))
                return true;
        }
        return false;
    }

    bool layer_priority_better(
        const LayerStats &lhs, const LayerStats &rhs) const {
        size_t lhs_net = layer_net_growth(lhs);
        size_t rhs_net = layer_net_growth(rhs);
        if (lhs_net != rhs_net)
            return lhs_net > rhs_net;
        size_t lhs_activity = lhs.recent_opened + lhs.recent_expanded;
        size_t rhs_activity = rhs.recent_opened + rhs.recent_expanded;
        if (lhs_activity != rhs_activity)
            return lhs_activity > rhs_activity;
        if (lhs.last_request_expansion != rhs.last_request_expansion)
            return lhs.last_request_expansion < rhs.last_request_expansion;
        return lhs.h < rhs.h;
    }

    int collect_layer_candidates(
        array<int, LAYER_TABLE_CAPACITY> &candidate_indices) const {
        int count = 0;
        for (int index = 0; index < LAYER_TABLE_CAPACITY; ++index) {
            if (layer_request_eligible(layers[index]))
                candidate_indices[count++] = index;
        }
        sort(candidate_indices.begin(), candidate_indices.begin() + count,
             [this](int lhs, int rhs) {
                 return layer_priority_better(layers[lhs], layers[rhs]);
             });
        return count;
    }

    bool global_stall_is_eligible() const {
        return config.enable_global_stall &&
               config.stall_expansions > 0 &&
               expansions_since_best_h >= config.stall_expansions &&
               !global_stall_requested;
    }

    int submit_layer_candidates(bool include_global_stall,
                                StateID fallback_state,
                                ap_float fallback_g,
                                ap_float fallback_h) {
        if (reached_request_limit() || request_spacing_active())
            return 0;

        array<int, LAYER_TABLE_CAPACITY> candidate_indices;
        int candidate_count = collect_layer_candidates(candidate_indices);
        int considered = min(candidate_count, config.candidate_layers);
        int submitted = 0;
        for (int position = 0;
             position < considered &&
             submitted < config.requests_per_slot;
             ++position) {
            LayerStats &layer = layers[candidate_indices[position]];
            string reason = include_global_stall && submitted == 0
                ? "frontier_growth_plateau+global_stall"
                : "frontier_growth_plateau";
            bool bypass_spacing = submitted > 0;
            size_t since_request_expanded =
                layer.expanded_since_request;
            size_t since_request_net_growth =
                layer_since_request_net_growth(layer);
            if (!request_state(
                    layer.representative_state, reason,
                    layer.representative_g, layer.h,
                    bypass_spacing)) {
                continue;
            }

            ++submitted;
            ++layer_requests_submitted;
            ++layer.requests_in_episode;
            layer.last_request_expansion = expansions;
            layer.opened_since_request = 0;
            layer.expanded_since_request = 0;
            cout << "[NLM-LLM-LAYER] selected"
                 << " h=" << layer.h
                 << " recent_opened=" << layer.recent_opened
                 << " recent_expanded=" << layer.recent_expanded
                 << " growth_ratio=" << layer_growth_ratio(layer)
                 << " since_request_expanded="
                 << since_request_expanded
                 << " since_request_net_growth="
                 << since_request_net_growth
                 << " episode_requests=" << layer.requests_in_episode
                 << " representative="
                 << state_id_label(layer.representative_state)
                 << endl;
        }

        if (include_global_stall && submitted == 0 &&
            fallback_state != StateID::no_state) {
            if (request_state(
                    fallback_state, "global_stall",
                    fallback_g, fallback_h)) {
                ++submitted;
            }
        }
        if (include_global_stall && submitted > 0)
            global_stall_requested = true;
        return submitted;
    }

    void analyze_layers(StateID state_id, ap_float g, ap_float h) {
        ++analysis_checks;
        for (LayerStats &layer : layers) {
            if (layer.occupied)
                update_layer_window(layer);
        }
        history_cursor = (history_cursor + 1) % config.activity_windows;

        bool global_stall = config.enable_global_stall &&
            config.stall_expansions > 0 &&
            expansions_since_best_h >= config.stall_expansions;
        if (global_stall && !global_stall_condition_active)
            ++global_stall_events;
        global_stall_condition_active = global_stall;

        submit_layer_candidates(
            global_stall_is_eligible(), state_id, g, h);
    }

    void log_heartbeat(StateID state_id, ap_float g, ap_float h) const {
        int tracked_layers = 0;
        int growing_layers = 0;
        int ready_layers = 0;
        const LayerStats *top_layer = nullptr;
        for (const LayerStats &layer : layers) {
            if (!layer.occupied)
                continue;
            ++tracked_layers;
            if (layer.growing) {
                ++growing_layers;
                if (!top_layer || layer_priority_better(layer, *top_layer))
                    top_layer = &layer;
            }
            if (layer_request_eligible(layer))
                ++ready_layers;
        }

        cout << "[NLM-LLM-MONITOR]"
             << " expansions=" << expansions
             << " opened=" << opened_states
             << " state=" << state_id_label(state_id)
             << " g=" << g
             << " h=" << h
             << " tracked_layers=" << tracked_layers
             << " growing_layers=" << growing_layers
             << " ready_layers=" << ready_layers;
        if (top_layer) {
            cout << " top_h=" << top_layer->h
                 << " top_opened=" << top_layer->recent_opened
                 << " top_expanded=" << top_layer->recent_expanded
                 << " top_growth_ratio="
                 << layer_growth_ratio(*top_layer)
                 << " top_since_request_expanded="
                 << top_layer->expanded_since_request
                 << " top_since_request_net_growth="
                 << layer_since_request_net_growth(*top_layer)
                 << " top_episode_requests="
                 << top_layer->requests_in_episode;
        }
        cout << " best_h=" << best_h
             << " stall_age=" << expansions_since_best_h
             << " submitted=" << requests_submitted
             << " pending=" << pending_states.size()
             << " layer_table_evictions=" << layer_table_evictions
             << " peak_memory_kb=" << utils::get_peak_memory_in_kb()
             << endl;
    }

public:
    LLMTriggerMonitor()
        : run_id(env_string("NLM_LLM_RUN_ID", "standalone")),
          anytime_iteration(1),
          next_request_id(0),
          expansions(0),
          expansions_since_best_h(0),
          last_analysis_expansion(0),
          last_ancestor_check_expansion(0),
          last_heartbeat_expansion(0),
          best_h(numeric_limits<ap_float>::infinity()),
          global_stall_condition_active(false),
          global_stall_requested(false),
          history_cursor(0),
          opened_states(0),
          analysis_checks(0),
          frontier_plateau_events(0),
          layer_episode_resets(0),
          layer_table_evictions(0),
          layer_requests_submitted(0),
          global_stall_events(0),
          ancestor_checks(0),
          ancestor_deferrals(0),
          ancestor_stagnation_events(0),
          request_attempts(0),
          requests_submitted(0),
          requests_rejected_duplicate(0),
          requests_rejected_pending_limit(0),
          requests_rejected_request_limit(0),
          requests_rejected_spacing(0),
          requests_rejected_bridge(0),
          responses_completed(0),
          response_transport_failures(0),
          usable_responses(0),
          injected_chains(0),
          injected_actions(0),
          injected_states(0),
          responses_discarded_phase_end(0),
          completed_unconsumed_phase_end(0),
          max_pending_observed(0),
          total_response_seconds(0.0),
          max_response_seconds(0.0),
          total_response_age_expansions(0),
          max_response_age_expansions(0),
          first_request_expansion(-1),
          last_request_expansion(-1),
          statistics_printed(false) {
        if (config.enabled) {
            cout << "[NLM-LLM-TRIGGER] enabled"
                 << " run_id=" << run_id
                 << " iteration=" << anytime_iteration
                 << " analysis_interval=" << config.analysis_interval
                 << " activity_windows=" << config.activity_windows
                 << " growth_confirm_windows="
                 << config.growth_confirm_windows
                 << " layer_reset_windows="
                 << config.layer_reset_windows
                 << " layer_min_recent_expanded="
                 << config.layer_min_recent_expanded
                 << " layer_min_recent_net_growth="
                 << config.layer_min_recent_net_growth
                 << " layer_min_since_request_expanded="
                 << config.layer_min_since_request_expanded
                 << " layer_min_since_request_net_growth="
                 << config.layer_min_since_request_net_growth
                 << " plateau_growth_ratio="
                 << config.plateau_growth_ratio
                 << " stall_expansions=" << config.stall_expansions
                 << " ancestor_check_interval="
                 << config.ancestor_check_interval
                 << " ancestor_depth=" << config.ancestor_depth
                 << " min_depth=" << config.min_depth
                 << " min_request_gap_expansions="
                 << config.min_request_gap_expansions
                 << " per_layer_request_gap_expansions="
                 << config.per_layer_request_gap_expansions
                 << " candidate_layers=" << config.candidate_layers
                 << " requests_per_slot=" << config.requests_per_slot
                 << " heartbeat_interval=" << config.heartbeat_interval
                 << " max_pending=" << config.max_pending
                 << " max_requests_per_iteration=" << config.max_requests
                 << " h_abs_epsilon=" << config.h_abs_epsilon
                 << " h_relative_epsilon=" << config.h_relative_epsilon
                 << " emit_state=" << (config.emit_state ? 1 : 0)
                 << " log_response_body="
                 << (config.log_response_body ? 1 : 0)
                 << " request_initial="
                 << (config.request_initial ? 1 : 0)
                 << " ancestor_trigger="
                 << (config.enable_ancestor_stagnation ? 1 : 0)
                 << " plateau_trigger="
                 << (config.enable_frontier_plateau ? 1 : 0)
                 << " global_stall_trigger="
                 << (config.enable_global_stall ? 1 : 0)
                 << " pending_behavior="
                 << (config.skip_pending ? "skip" : "normal")
                 << endl;
        }
    }

    ~LLMTriggerMonitor() {
        finalize_and_print();
    }

    void finalize_and_print() {
        if (statistics_printed)
            return;
        statistics_printed = true;
        bridge.stop();
        completed_unconsumed_phase_end = bridge.completed_unconsumed();
        poll_bridge(true);
        if (!config.enabled)
            return;
        double average_response_seconds = responses_completed > 0
            ? total_response_seconds / responses_completed : 0.0;
        double average_response_age = responses_completed > 0
            ? static_cast<double>(total_response_age_expansions) /
                  responses_completed
            : 0.0;
        double average_request_gap = requests_submitted > 1
            ? static_cast<double>(
                  last_request_expansion - first_request_expansion) /
                  (requests_submitted - 1)
            : 0.0;
        cout << "[NLM-LLM-TRIGGER-STATS]"
             << " run_id=" << run_id
             << " iteration=" << anytime_iteration
             << " expansions=" << expansions
             << " opened=" << opened_states
             << " analysis_checks=" << analysis_checks
             << " plateau_events=" << frontier_plateau_events
             << " layer_episode_resets=" << layer_episode_resets
             << " layer_table_evictions=" << layer_table_evictions
             << " layer_requests=" << layer_requests_submitted
             << " global_stall_events=" << global_stall_events
             << " ancestor_checks=" << ancestor_checks
             << " ancestor_deferrals=" << ancestor_deferrals
             << " ancestor_events=" << ancestor_stagnation_events
             << " request_attempts=" << request_attempts
             << " submitted=" << requests_submitted
             << " rejected_duplicate=" << requests_rejected_duplicate
             << " rejected_pending_limit="
             << requests_rejected_pending_limit
             << " rejected_request_limit="
             << requests_rejected_request_limit
             << " rejected_spacing=" << requests_rejected_spacing
             << " request_limit_reached="
             << (reached_request_limit() ? 1 : 0)
             << " first_request_expansion=" << first_request_expansion
             << " last_request_expansion=" << last_request_expansion
             << " avg_request_gap_expansions=" << average_request_gap
             << " rejected_bridge=" << requests_rejected_bridge
             << " responses=" << responses_completed
             << " transport_failures=" << response_transport_failures
             << " usable_responses=" << usable_responses
             << " injected_chains=" << injected_chains
             << " injected_actions=" << injected_actions
             << " injected_states=" << injected_states
             << " discarded_phase_end=" << responses_discarded_phase_end
             << " completed_unconsumed="
             << completed_unconsumed_phase_end
             << " discarded_queued=" << bridge.queued_discarded()
             << " cancelled_inflight=" << bridge.active_cancelled()
             << " max_pending=" << max_pending_observed
             << " avg_response_seconds=" << average_response_seconds
             << " max_response_seconds=" << max_response_seconds
             << " avg_response_age_expansions=" << average_response_age
             << " max_response_age_expansions="
             << max_response_age_expansions
             << " peak_memory_kb=" << utils::get_peak_memory_in_kb()
             << endl;
        for (const auto &entry : request_attempts_by_reason) {
            const string &reason = entry.first;
            cout << "[NLM-LLM-TRIGGER-REASON-STATS]"
                 << " reason=\"" << reason << "\""
                 << " attempts=" << entry.second
                 << " submitted=" << requests_submitted_by_reason[reason]
                 << " responses=" << responses_completed_by_reason[reason]
                 << endl;
        }
    }

    bool enabled() const {
        return config.enabled;
    }

    void start_bridge() {
        if (config.enabled)
            bridge.start();
    }

    void set_anytime_iteration(int iteration) {
        anytime_iteration = max(1, iteration);
    }

    void record_usable_response() {
        ++usable_responses;
    }

    void record_injected_chain(int applied_actions, int inserted_count) {
        if (applied_actions <= 0)
            return;
        ++injected_chains;
        injected_actions += static_cast<size_t>(applied_actions);
        injected_states += static_cast<size_t>(max(0, inserted_count));
    }

    vector<LLMResponse> poll_bridge(bool discard_for_phase_end = false) {
        if (!config.enabled)
            return vector<LLMResponse>();
        vector<LLMResponse> responses = bridge.poll_completed();
        for (const LLMResponse &response : responses) {
            if (discard_for_phase_end) {
                ++responses_discarded_phase_end;
                pending_states.erase(response.state_id);
                pending_request_infos.erase(response.state_id);
                cout << "[NLM-LLM-BRIDGE] discarded"
                     << " run_id=" << run_id
                     << " iteration=" << anytime_iteration
                     << " request_id=" << response.request_id
                     << " state=" << response.state_label
                     << " reason=phase_end"
                     << endl;
                continue;
            }
            ++responses_completed;
            if (!response.transport_ok)
                ++response_transport_failures;
            double response_seconds = 0.0;
            size_t response_age_expansions = 0;
            string request_reason;
            auto request_it = pending_request_infos.find(response.state_id);
            if (request_it != pending_request_infos.end()) {
                response_seconds =
                    chrono::duration_cast<chrono::duration<double>>(
                        chrono::steady_clock::now() -
                        request_it->second.submitted_at)
                        .count();
                response_age_expansions = static_cast<size_t>(max(
                    0, expansions - request_it->second.expansions_at_submit));
                request_reason = request_it->second.reason;
                ++responses_completed_by_reason[request_reason];
                pending_request_infos.erase(request_it);
            }
            total_response_seconds += response_seconds;
            max_response_seconds = max(max_response_seconds, response_seconds);
            total_response_age_expansions += response_age_expansions;
            max_response_age_expansions = max(
                max_response_age_expansions, response_age_expansions);
            pending_states.erase(response.state_id);
            cout << "[NLM-LLM-BRIDGE] completed"
                 << " request_id=" << response.request_id
                 << " state=" << response.state_label
                 << " transport_ok=" << (response.transport_ok ? 1 : 0)
                 << " http_status=" << response.http_status
                 << " body_bytes=" << response.body.size()
                 << " latency_seconds=" << response_seconds
                 << " age_expansions=" << response_age_expansions;
            if (!request_reason.empty())
                cout << " reason=" << request_reason;
            if (!response.error.empty())
                cout << " error=\"" << response.error << "\"";
            cout << endl;
            if (config.log_response_body && !response.body.empty()) {
                cout << "[NLM-LLM-BRIDGE-RESPONSE] begin"
                     << " request_id=" << response.request_id
                     << " state=" << response.state_label
                     << endl;
                cout << response.body << endl;
                cout << "[NLM-LLM-BRIDGE-RESPONSE] end"
                     << " request_id=" << response.request_id
                     << " state=" << response.state_label
                     << endl;
            }
        }
        return responses;
    }

    bool has_pending_requests() const {
        return !pending_states.empty();
    }

    bool skips_pending_states() const {
        return config.skip_pending;
    }

    bool maybe_request_initial(StateID state_id, ap_float g, ap_float h) {
        if (!config.request_initial)
            return false;
        return request_state(
            state_id, "initial_replay_test", g, h, true);
    }

    void record_open_state(ap_float h) {
        if (!config.enabled)
            return;
        if (!std::isfinite(static_cast<double>(h)))
            return;

        ++opened_states;
        if (config.enable_frontier_plateau) {
            LayerStats &layer = layer_for_h(h);
            ++layer.current_opened;
            ++layer.opened_since_request;
            layer.last_seen_expansion = expansions;
        }

        if (meaningfully_improves_h(best_h, h)) {
            best_h = h;
            expansions_since_best_h = 0;
            global_stall_condition_active = false;
            global_stall_requested = false;
        }
    }

    void record_frontier_reinsert(ap_float h) {
        record_open_state(h);
    }

    bool suspend_if_pending(StateID state_id) {
        if (!config.enabled || !config.skip_pending ||
            !pending_states.count(state_id)) {
            return false;
        }
        suspended_states.insert(state_id);
        return true;
    }

    bool take_suspended(StateID state_id) {
        return suspended_states.erase(state_id) > 0;
    }

    bool should_check_ancestor() {
        if (!config.enabled || !config.enable_ancestor_stagnation)
            return false;
        if (reached_request_limit())
            return false;
        if (request_spacing_active())
            return false;
        if (expansions - last_ancestor_check_expansion <
            config.ancestor_check_interval) {
            return false;
        }
        last_ancestor_check_expansion = expansions;
        // A mature frontier layer or global stall gets the next shared
        // request slot. It will be submitted at the next sparse analysis;
        // do not let the fixed ancestor sampling instant preempt it.
        if (has_eligible_layer_candidate() || global_stall_is_eligible()) {
            ++ancestor_deferrals;
            return false;
        }
        ++ancestor_checks;
        return true;
    }

    int ancestor_depth() const {
        return config.ancestor_depth;
    }

    int ancestor_min_depth() const {
        return config.min_depth;
    }

    bool ancestor_improvement_is_stagnant(
        ap_float best_improvement, ap_float current_h) const {
        return best_improvement <= h_improvement_threshold(
            current_h + best_improvement, current_h);
    }

    bool request_ancestor(StateID state_id, ap_float g, ap_float h) {
        ++ancestor_stagnation_events;
        return request_state(state_id, "ancestor_stagnation", g, h);
    }

    void record_expanded(StateID state_id, ap_float g, ap_float h) {
        if (!config.enabled)
            return;
        if (!std::isfinite(static_cast<double>(h))) {
            record_expanded_without_h();
            return;
        }
        ++expansions;
        ++expansions_since_best_h;
        if (config.enable_frontier_plateau) {
            LayerStats &layer = layer_for_h(h);
            ++layer.current_expanded;
            ++layer.expanded_since_request;
            layer.h = h;
            layer.last_seen_expansion = expansions;
            layer.representative_state = state_id;
            layer.representative_g = g;
            layer.representative_expansion = expansions;
        }

        if (expansions - last_analysis_expansion >=
            config.analysis_interval) {
            last_analysis_expansion = expansions;
            analyze_layers(state_id, g, h);
        }
        if (config.heartbeat_interval > 0 &&
            expansions - last_heartbeat_expansion >=
                config.heartbeat_interval) {
            last_heartbeat_expansion = expansions;
            log_heartbeat(state_id, g, h);
        }
    }

    void record_expanded_without_h() {
        if (!config.enabled)
            return;
        ++expansions;
        ++expansions_since_best_h;
    }
};

EagerSearch::EagerSearch(const Options &opts)
    : SearchEngine(opts),
      reopen_closed_nodes(opts.get<bool>("reopen_closed")),
      use_multi_path_dependence(opts.get<bool>("mpd")),
      llm_h_open_list_key_index(
          opts.get<int>("llm_h_open_list_key_index", -1)),
      llm_h_evaluator(opts.get<Heuristic *>("llm_h", nullptr)),
      open_list(opts.get<shared_ptr<OpenListFactory>>("open")->
                create_state_open_list()),
      f_evaluator(opts.get<ScalarEvaluator *>("f_eval", nullptr)),
      preferred_operator_heuristics(opts.get_list<Heuristic *>("preferred")),
      pruning_method(opts.get<shared_ptr<PruningMethod>>("pruning")),
      llm_trigger_monitor(new LLMTriggerMonitor()) {
}

EagerSearch::~EagerSearch() = default;

void EagerSearch::set_anytime_iteration(int iteration) {
    SearchEngine::set_anytime_iteration(iteration);
    llm_trigger_monitor->set_anytime_iteration(iteration);
}

void EagerSearch::initialize() {
    cout << "Conducting best first search"
         << (reopen_closed_nodes ? " with" : " without")
         << " reopening closed nodes, (real) bound = " << bound
         << endl;
    if (use_multi_path_dependence)
        cout << "Using multi-path dependence (LM-A*)" << endl;
    assert(open_list);
    if (PLAN_VIS_LOG == latex_only) {
    	g_plan_logger->register_latex_var("x");
    	g_plan_logger->register_latex_var("y");
    }

    set<Heuristic *> hset;
    open_list->get_involved_heuristics(hset);

    // add heuristics that are used for preferred operators (in case they are
    // not also used in the open list)
    hset.insert(preferred_operator_heuristics.begin(),
                preferred_operator_heuristics.end());

    // add heuristics that are used in the f_evaluator. They are usually also
    // used in the open list and hence already be included, but we want to be
    // sure.
    if (f_evaluator) {
        f_evaluator->get_involved_heuristics(hset);
    }

    heuristics.assign(hset.begin(), hset.end());
    assert(!heuristics.empty());

    // get_involved_heuristics returns a set, whose order is unrelated to the
    // lexicographic Open List keys. Keep legacy single-heuristic searches
    // working, but require multi-heuristic online configurations to identify
    // the h evaluator explicitly instead of silently choosing goalcount (or
    // another secondary evaluator) by pointer order.
    if (!llm_h_evaluator) {
        llm_h_evaluator = heuristics[0];
        if (llm_trigger_monitor->enabled() && heuristics.size() > 1) {
            cout << "[NLM-LLM-TRIGGER] warning"
                 << " reason=ambiguous_llm_h_evaluator"
                 << " detail=configure_llm_h"
                 << endl;
        }
    }

    if (llm_trigger_monitor->enabled()) {
        if (llm_h_open_list_key_index < 0 &&
            !use_multi_path_dependence) {
            cout << "[NLM-LLM-TRIGGER] warning"
                 << " reason=open_list_h_key_unavailable"
                 << " detail=configure_llm_h_open_list_key_index"
                 << endl;
        }
        llm_action_chain_evaluator.reset(new ActionChainEvaluator());
        llm_trigger_monitor->start_bridge();
    }

    const GlobalState &initial_state = g_initial_state();
    // Note: we consider the initial state as reached by a preferred
    // operator.
    EvaluationContext eval_context(initial_state, 0, true, &statistics);

    statistics.inc_evaluated_states();

    if (open_list->is_dead_end(eval_context)) {
        cout << "Initial state is a dead end." << endl;
    } else {
        if (search_progress.check_progress(eval_context))
            print_checkpoint_line(0);
        start_f_value_statistics(eval_context);
        SearchNode node = search_space.get_node(initial_state);
        node.open_initial();

        open_list->insert(eval_context, initial_state.get_id());
        if (llm_trigger_monitor->enabled()) {
            ap_float initial_h =
                eval_context.get_heuristic_value(llm_h_evaluator);
            llm_trigger_monitor->record_open_state(initial_h);
            llm_trigger_monitor->maybe_request_initial(
                initial_state.get_id(), 0, initial_h);
        }
    }

    print_initial_h_values(eval_context);
    if (PLAN_VIS_LOG == plan_vis_log) {
        utils::Timer h_time;
        if (PLAN_VIS_LOG) h_time.reset();
        ap_float h_val = eval_context.get_heuristic_value(llm_h_evaluator);
        if (PLAN_VIS_LOG) {
        	h_time.stop();
        	g_plan_logger->log_node(
        			initial_state.get_id(),
					initial_state.dump_plan_vis_log(),
					h_val,
					h_time,
					search_space.get_node(initial_state).get_g(),
					initial_state.get_id(),
					test_goal(initial_state),
					true);
        }
    }
}

void EagerSearch::print_checkpoint_line(int g) const {
    cout << "[g=" << g << ", ";
    statistics.print_basic_statistics();
    cout << "]" << endl;
}

void EagerSearch::print_statistics() const {
    statistics.print_detailed_statistics();
    search_space.print_statistics();
    pruning_method->print_statistics();
    llm_trigger_monitor->finalize_and_print();
}

bool EagerSearch::llm_ancestor_stagnant(
    const SearchNode &node, ap_float current_h) {
    // Ancestor checks are intentionally sparse. Reuse the native parent
    // pointers instead of maintaining a second per-state ancestry table.
    int comparison_depth = llm_trigger_monitor->ancestor_depth();
    int minimum_path_depth = llm_trigger_monitor->ancestor_min_depth();
    int traversal_depth = max(comparison_depth, minimum_path_depth);
    StateID ancestor_id = node.get_parent_state_id();
    ap_float best_improvement = 0;
    int inspected = 0;

    while (ancestor_id != StateID::no_state &&
           inspected < traversal_depth) {
        GlobalState ancestor_state =
            g_state_registry->lookup_state(ancestor_id);
        SearchNode ancestor_node = search_space.get_node(ancestor_state);
        if (inspected < comparison_depth) {
            EvaluationContext ancestor_context(
                ancestor_state, ancestor_node.get_g(), false, nullptr);
            if (ancestor_context.is_heuristic_infinite(llm_h_evaluator))
                return false;
            ap_float ancestor_h =
                ancestor_context.get_heuristic_value(llm_h_evaluator);
            best_improvement = max(
                best_improvement, ancestor_h - current_h);
        }
        ancestor_id = ancestor_node.get_parent_state_id();
        ++inspected;
    }

    return inspected >= traversal_depth &&
           llm_trigger_monitor->ancestor_improvement_is_stagnant(
               best_improvement, current_h);
}

void EagerSearch::requeue_llm_source(StateID source_id) {
    if (!llm_trigger_monitor->skips_pending_states())
        return;

    GlobalState source_state = g_state_registry->lookup_state(source_id);
    SearchNode source_node = search_space.get_node(source_state);
    if (!source_node.is_open() || source_node.is_dead_end())
        return;

    EvaluationContext eval_context(
        source_state, source_node.get_g(), false, &statistics);
    if (open_list->is_dead_end(eval_context)) {
        source_node.mark_as_dead_end();
        statistics.inc_dead_ends();
        return;
    }
    open_list->insert(eval_context, source_id);
    ap_float source_h = eval_context.get_heuristic_value(llm_h_evaluator);
    llm_trigger_monitor->record_frontier_reinsert(source_h);
    cout << "[NLM-LLM-INJECT] requeued source="
         << state_id_label(source_id)
         << " reason=resume_after_llm"
         << endl;
}

bool EagerSearch::inject_llm_action_chain(
    const string &request_id, StateID source_id,
    const vector<string> &actions) {
    GlobalState current_state = g_state_registry->lookup_state(source_id);
    int applied_actions = 0;
    int inserted_states = 0;

    for (const string &raw_action : actions) {
        assert(llm_action_chain_evaluator);
        ActionResolution resolution =
            llm_action_chain_evaluator->resolve_action(
                current_state, raw_action);
        if (resolution.status == ActionResolutionStatus::UNKNOWN_ACTION) {
            cout << "[NLM-LLM-INJECT] unknown action=\"" << raw_action
                 << "\" source=" << state_id_label(source_id) << endl;
            break;
        }
        if (resolution.status ==
            ActionResolutionStatus::INAPPLICABLE_ACTION) {
            cout << "[NLM-LLM-INJECT] action no longer applicable=\""
                 << raw_action << "\" state=" << current_state.get_id()
                 << endl;
            break;
        }
        const GlobalOperator *selected_operator = resolution.op;

        SearchNode parent_node = search_space.get_node(current_state);
        if (parent_node.is_dead_end() ||
            parent_node.get_real_g() + selected_operator->get_cost() >= bound) {
            break;
        }

        GlobalState successor_state =
            llm_action_chain_evaluator->apply_action(
                current_state, *selected_operator);
        statistics.inc_generated();
        SearchNode successor_node = search_space.get_node(successor_state);
        if (successor_node.is_dead_end())
            break;
        bool successor_was_new = successor_node.is_new();

        if (use_multi_path_dependence || successor_was_new) {
            for (Heuristic *heuristic : heuristics) {
                heuristic->reach_state(
                    current_state, *selected_operator, successor_state);
            }
        }

        ap_float successor_g =
            parent_node.get_g() + get_adjusted_cost(*selected_operator);
        if (successor_was_new) {
            EvaluationContext eval_context(
                successor_state, successor_g, false, &statistics);
            statistics.inc_evaluated_states();
            if (open_list->is_dead_end(eval_context)) {
                successor_node.mark_as_dead_end();
                statistics.inc_dead_ends();
                break;
            }

            ap_float successor_h =
                eval_context.get_heuristic_value(llm_h_evaluator);
            successor_node.open(parent_node, selected_operator);
            open_list->insert(eval_context, successor_state.get_id());
            llm_trigger_monitor->record_open_state(successor_h);
            ++inserted_states;

            if (search_progress.check_progress(eval_context)) {
                print_checkpoint_line(successor_node.get_g());
                reward_progress();
            }
        } else if (successor_node.get_g() > successor_g) {
            if (reopen_closed_nodes) {
                if (successor_node.is_closed())
                    statistics.inc_reopened();
                successor_node.reopen(parent_node, selected_operator);
                EvaluationContext eval_context(
                    successor_state, successor_node.get_g(),
                    false, &statistics);
                open_list->insert(eval_context, successor_state.get_id());
                ap_float successor_h =
                    eval_context.get_heuristic_value(llm_h_evaluator);
                llm_trigger_monitor->record_open_state(successor_h);
                ++inserted_states;
            } else {
                successor_node.update_parent(parent_node, selected_operator);
            }
        }

        ++applied_actions;
        cout << "[NLM-LLM-INJECT] action=" << raw_action
             << " parent=" << current_state.get_id()
             << " successor=" << successor_state.get_id()
             << " inserted=" << (successor_was_new ? 1 : 0)
             << endl;
        current_state = successor_state;
    }

    llm_trigger_monitor->record_injected_chain(
        applied_actions, inserted_states);
    cout << "[NLM-LLM-INJECT] chain"
         << " request_id=" << request_id
         << " source=" << state_id_label(source_id)
         << " requested_actions=" << actions.size()
         << " applied_actions=" << applied_actions
         << " inserted_states=" << inserted_states
         << endl;
    return applied_actions > 0;
}

void EagerSearch::poll_llm_responses() {
    vector<LLMResponse> responses = llm_trigger_monitor->poll_bridge();
    for (const LLMResponse &response : responses) {
        bool source_was_suspended =
            llm_trigger_monitor->take_suspended(response.state_id);

        if (!response.transport_ok || response.body.empty()) {
            // The bridge already logged the transport failure. If the source
            // was never popped, its existing Open List entry remains valid.
        } else {
            ParsedLLMResponse parsed = parse_llm_response_body(response.body);
            if (!parsed.valid) {
                cout << "[NLM-LLM-INJECT] response parse error"
                     << " request_id=" << response.request_id
                     << " error=\"" << parsed.error << "\""
                     << endl;
            } else {
                size_t total_actions = 0;
                for (const vector<string> &chain : parsed.action_chains)
                    total_actions += chain.size();
                cout << "[NLM-LLM-INJECT] response"
                     << " request_id=" << response.request_id
                     << " state=" << response.state_label
                     << " status=" << parsed.status
                     << " samples=" << parsed.action_chains.size()
                     << " actions=" << total_actions
                     << endl;
                bool usable_status =
                    parsed.status == "ok" || parsed.status == "partial";
                if (usable_status && total_actions > 0)
                    llm_trigger_monitor->record_usable_response();
                if (usable_status) {
                    for (size_t sample_index = 0;
                         sample_index < parsed.action_chains.size();
                         ++sample_index) {
                        const vector<string> &chain =
                            parsed.action_chains[sample_index];
                        cout << "[NLM-LLM-INJECT] sample"
                             << " request_id=" << response.request_id
                             << " sample=" << sample_index
                             << " actions=" << chain.size()
                             << endl;
                        if (!chain.empty())
                            inject_llm_action_chain(
                                response.request_id,
                                response.state_id,
                                chain);
                    }
                } else if (total_actions > 0) {
                    cout << "[NLM-LLM-INJECT] ignored actions for status="
                         << parsed.status
                         << " request_id=" << response.request_id
                         << endl;
                }
            }
        }

        // A pending source is removed from the Open List when popped. Resume
        // exactly those sources after completion, independent of response
        // latency or whether the LLM prefix was usable.
        if (source_was_suspended)
            requeue_llm_source(response.state_id);
    }
}

SearchStatus EagerSearch::step() {
    pair<SearchNode, bool> n = fetch_next_node();
    if (!n.second) {
        return FAILED;
    }
    SearchNode node = n.first;

    GlobalState s = node.get_state();

	if (PLAN_VIS_LOG == latex_only) {
		g_plan_logger->log_latex_explored(s.get_numeric_state_vals_string());
	}

    if (check_goal_and_set_plan(s)) {
        llm_trigger_monitor->poll_bridge();
        return SOLVED;
    }

    vector<const GlobalOperator *> applicable_ops;
    set<const GlobalOperator *> preferred_ops;

    g_successor_generator->generate_applicable_ops(s, applicable_ops);

    /*
      TODO: When preferred operators are in use, a preferred operator will be
      considered by the preferred operator queues even when it is pruned.
    */
    pruning_method->prune_operators(s, applicable_ops);

    // This evaluates the expanded state (again) to get preferred ops
    EvaluationContext eval_context(s, node.get_g(), false, &statistics, true);
    for (Heuristic *heur : preferred_operator_heuristics) {
        /* In an alternation search with unreliable heuristics, it is
           possible that this heuristic considers the state a dead
           end. We only want to ask for preferred operators for
           finite-value heuristics. */
        if (!eval_context.is_heuristic_infinite(heur)) {
            vector<const GlobalOperator *> preferred =
                eval_context.get_preferred_operators(heur);
            preferred_ops.insert(preferred.begin(), preferred.end());
        }
    }

    for (const GlobalOperator *op : applicable_ops) {
        if ((node.get_real_g() + op->get_cost()) >= bound)
            continue;

        GlobalState succ_state = g_state_registry->get_successor_state(s, *op);
        statistics.inc_generated();
        bool is_preferred = (preferred_ops.find(op) != preferred_ops.end());

        SearchNode succ_node = search_space.get_node(succ_state);

        // Previously encountered dead end. Don't re-evaluate.
        if (succ_node.is_dead_end())
            continue;

        // update new path
        if (use_multi_path_dependence || succ_node.is_new()) {
            /*
              Note: we must call reach_state for each heuristic, so
              don't break out of the for loop early.
            */
            for (Heuristic *heuristic : heuristics) {
                heuristic->reach_state(s, *op, succ_state);
            }
        }

        if (succ_node.is_new()) {
            // We have not seen this state before.
            // Evaluate and create a new node.

            // Careful: succ_node.get_g() is not available here yet,
            // hence the stupid computation of succ_g.
            // TODO: Make this less fragile.
            ap_float succ_g = node.get_g() + get_adjusted_cost(*op);

            EvaluationContext eval_context(
                succ_state, succ_g, is_preferred, &statistics);
            statistics.inc_evaluated_states();

            if (open_list->is_dead_end(eval_context)) {
                succ_node.mark_as_dead_end();
                statistics.inc_dead_ends();
                continue;
            }
            utils::Timer h_time;
            if (PLAN_VIS_LOG == plan_vis_log) {h_time.reset();
            assert(false); // todo: code changed here, have to fetch time somewhere else
			}
//            ap_float succ_h = 4;
//  was before:
            ap_float succ_h =
                eval_context.get_heuristic_value(llm_h_evaluator);
            if (PLAN_VIS_LOG == plan_vis_log) h_time.stop();

            succ_node.open(node, op);

            open_list->insert(eval_context, succ_state.get_id());
            llm_trigger_monitor->record_open_state(succ_h);
            if (search_progress.check_progress(eval_context)) {
                print_checkpoint_line(succ_node.get_g());
                reward_progress();
            }
            if (PLAN_VIS_LOG == plan_vis_log) {
            	g_plan_logger->log_node(
            			succ_state.get_id(),
						succ_state.dump_plan_vis_log(s),
						succ_h,
    					h_time,
						succ_g,
						s.get_id(),
						test_goal(succ_state),
    					false);
            }
        } else if (succ_node.get_g() > node.get_g() + get_adjusted_cost(*op)) {
            // We found a new cheapest path to an open or closed state.
            if (reopen_closed_nodes) {
                if (succ_node.is_closed()) {
                    /*
                      TODO: It would be nice if we had a way to test
                      that reopening is expected behaviour, i.e., exit
                      with an error when this is something where
                      reopening should not occur (e.g. A* with a
                      consistent heuristic).
                    */
                    statistics.inc_reopened();
                }
                succ_node.reopen(node, op);

                EvaluationContext eval_context(
                    succ_state, succ_node.get_g(), is_preferred, &statistics);

                /*
                  Note: our old code used to retrieve the h value from
                  the search node here. Our new code recomputes it as
                  necessary, thus avoiding the incredible ugliness of
                  the old "set_evaluator_value" approach, which also
                  did not generalize properly to settings with more
                  than one heuristic.

                  Reopening should not happen all that frequently, so
                  the performance impact of this is hopefully not that
                  large. In the medium term, we want the heuristics to
                  remember heuristic values for states themselves if
                  desired by the user, so that such recomputations
                  will just involve a look-up by the Heuristic object
                  rather than a recomputation of the heuristic value
                  from scratch.
                */
                open_list->insert(eval_context, succ_state.get_id());
                if (llm_trigger_monitor->enabled()) {
                    ap_float reopen_h =
                        eval_context.get_heuristic_value(llm_h_evaluator);
                    llm_trigger_monitor->record_open_state(reopen_h);
                }
            } else {
                // If we do not reopen closed nodes, we just update the parent pointers.
                // Note that this could cause an incompatibility between
                // the g-value and the actual path that is traced back.
                succ_node.update_parent(node, op);
            }
            if (PLAN_VIS_LOG == plan_vis_log)
            	g_plan_logger->log_duplicate(succ_state.get_id(),node.get_g() + get_adjusted_cost(*op),s.get_id());
        }
    }

    poll_llm_responses();
    return IN_PROGRESS;
}

pair<SearchNode, bool> EagerSearch::fetch_next_node() {
    /* TODO: The bulk of this code deals with multi-path dependence,
       which is a bit unfortunate since that is a special case that
       makes the common case look more complicated than it would need
       to be. We could refactor this by implementing multi-path
       dependence as a separate search algorithm that wraps the "usual"
       search algorithm and adds the extra processing in the desired
       places. I think this would lead to much cleaner code. */

    while (true) {
        poll_llm_responses();
        if (open_list->empty()) {
            if (llm_trigger_monitor->has_pending_requests()) {
                this_thread::sleep_for(chrono::milliseconds(10));
                continue;
            }
            cout << "Completely explored state space -- no solution!" << endl;
            // HACK! HACK! we do this because SearchNode has no default/copy constructor
            SearchNode dummy_node = search_space.get_node(g_initial_state());
            return make_pair(dummy_node, false);
        }
        // Reuse this small buffer across calls. Online-search configurations
        // explicitly identify which Open List key component is the primary h
        // value: single-evaluator greedy uses index 0, A* uses index 1 in
        // [f, h], and lexicographic satisfying search uses index 0 in
        // [h, goalcount]. This avoids heuristic recomputation and per-step
        // allocation while allowing arbitrary secondary tie breakers.
        removed_key_buffer.clear();
        bool capture_removed_key = use_multi_path_dependence ||
            (llm_trigger_monitor->enabled() &&
             llm_h_open_list_key_index >= 0);
        StateID id = open_list->remove_min(
            capture_removed_key ? &removed_key_buffer : nullptr);
        // TODO is there a way we can avoid creating the state here and then
        //      recreate it outside of this function with node.get_state()?
        //      One way would be to store GlobalState objects inside SearchNodes
        //      instead of StateIDs
        GlobalState s = g_state_registry->lookup_state(id);
        if (violates_global_constraint(s)) continue;
        SearchNode node = search_space.get_node(s);

        if (node.is_closed())
            continue;
        if (llm_trigger_monitor->suspend_if_pending(id))
            continue;

        if (use_multi_path_dependence) {
            assert(removed_key_buffer.size() == 2);
            if (node.is_dead_end())
                continue;
            ap_float pushed_h = removed_key_buffer[1];

            if (!node.is_closed()) {
                EvaluationContext eval_context(
                    node.get_state(), node.get_g(), false, &statistics);

                if (open_list->is_dead_end(eval_context)) {
                    node.mark_as_dead_end();
                    statistics.inc_dead_ends();
                    continue;
                }
                if (pushed_h < eval_context.get_result(heuristics[0]).get_h_value()) {
                    assert(node.is_open());
                    open_list->insert(eval_context, node.get_state_id());
                    if (llm_trigger_monitor->enabled()) {
                        ap_float current_h =
                            eval_context.get_result(
                                llm_h_evaluator).get_h_value();
                        llm_trigger_monitor->record_frontier_reinsert(
                            current_h);
                    }
                    continue;
                }
            }
        }

        node.close();
        assert(!node.is_dead_end());
        update_f_value_statistics(node);
        statistics.inc_expanded();
        bool captured_llm_h = llm_h_open_list_key_index >= 0 &&
            static_cast<size_t>(llm_h_open_list_key_index) <
                removed_key_buffer.size();
        if (captured_llm_h) {
            ap_float popped_h =
                removed_key_buffer[llm_h_open_list_key_index];
            llm_trigger_monitor->record_expanded(
                id, node.get_g(), popped_h);
            if (llm_trigger_monitor->should_check_ancestor() &&
                llm_ancestor_stagnant(node, popped_h)) {
                llm_trigger_monitor->request_ancestor(
                    id, node.get_g(), popped_h);
            }
        } else {
            llm_trigger_monitor->record_expanded_without_h();
        }
        return make_pair(node, true);
    }
}

void EagerSearch::reward_progress() {
    // Boost the "preferred operator" open lists somewhat whenever
    // one of the heuristics finds a state with a new best h value.
    open_list->boost_preferred();
}

void EagerSearch::dump_search_space() const {
    search_space.dump();
}

void EagerSearch::start_f_value_statistics(EvaluationContext &eval_context) {
    if (f_evaluator) {
        ap_float f_value = eval_context.get_heuristic_value(f_evaluator);
        statistics.report_f_value_progress(f_value);
    }
}

/* TODO: HACK! This is very inefficient for simply looking up an h value.
   Also, if h values are not saved it would recompute h for each and every state. */
void EagerSearch::update_f_value_statistics(const SearchNode &node) {
    if (f_evaluator) {
        /*
          TODO: This code doesn't fit the idea of supporting
          an arbitrary f evaluator.
        */
        EvaluationContext eval_context(node.get_state(), node.get_g(), false, &statistics);
        ap_float f_value = eval_context.get_heuristic_value(f_evaluator);
        statistics.report_f_value_progress(f_value);
    }
}

/* TODO: merge this into SearchEngine::add_options_to_parser when all search
         engines support pruning. */
void add_pruning_option(OptionParser &parser) {
    parser.add_option<shared_ptr<PruningMethod>>(
        "pruning",
        "Pruning methods can prune or reorder the set of applicable operators in "
        "each state and thereby influence the number and order of successor states "
        "that are considered.",
        "null()");
}

static SearchEngine *_parse(OptionParser &parser) {
    parser.document_synopsis("Eager best-first search", "");

    parser.add_option<shared_ptr<OpenListFactory>>("open", "open list");
    parser.add_option<bool>("reopen_closed",
                            "reopen closed nodes", "false");
    parser.add_option<ScalarEvaluator *>(
        "f_eval",
        "set evaluator for jump statistics. "
        "(Optional; if no evaluator is used, jump statistics will not be displayed.)",
        OptionParser::NONE);
    parser.add_list_option<Heuristic *>(
        "preferred",
        "use preferred operators of these heuristics", "[]");
    parser.add_option<int>(
        "llm_h_open_list_key_index",
        "zero-based Open List key component containing the h value used by "
        "the LLM trigger monitor; -1 disables popped-h capture",
        "-1");
    parser.add_option<Heuristic *>(
        "llm_h",
        "heuristic used for LLM trigger bookkeeping, ancestry checks and "
        "injected-state evaluation; configure this explicitly when the Open "
        "List contains more than one heuristic",
        OptionParser::NONE);

    add_pruning_option(parser);
    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();

    EagerSearch *engine = nullptr;
    if (!parser.dry_run()) {
        opts.set<bool>("mpd", false);
        engine = new EagerSearch(opts);
    }

    return engine;
}

static SearchEngine *_parse_astar(OptionParser &parser) {
    parser.document_synopsis(
        "A* search (eager)",
        "A* is a special case of eager best first search that uses g+h "
        "as f-function. "
        "We break ties using the evaluator. Closed nodes are re-opened.");
    parser.document_note(
        "mpd option",
        "This option is currently only present for the A* algorithm and not "
        "for the more general eager search, "
        "because the current implementation of multi-path depedence "
        "does not support general open lists.");
    parser.document_note(
        "Equivalent statements using general eager search",
        "\n```\n--search astar(evaluator)\n```\n"
        "is equivalent to\n"
        "```\n--heuristic h=evaluator\n"
        "--search eager(tiebreaking([sum([g(), h]), h], unsafe_pruning=false),\n"
        "               reopen_closed=true, f_eval=sum([g(), h]))\n"
        "```\n", true);
    parser.add_option<ScalarEvaluator *>("eval", "evaluator for h-value");
    parser.add_option<bool>("mpd",
                            "use multi-path dependence (LM-A*)", "false");

    add_pruning_option(parser);
    SearchEngine::add_options_to_parser(parser);
    Options opts = parser.parse();

    EagerSearch *engine = nullptr;
    if (!parser.dry_run()) {
        auto temp = search_common::create_astar_open_list_factory_and_f_eval(opts);
        opts.set("open", temp.first);
        opts.set("f_eval", temp.second);
        opts.set("reopen_closed", true);
        opts.set("llm_h_open_list_key_index", 1);
        vector<Heuristic *> preferred_list;
        opts.set("preferred", preferred_list);
        engine = new EagerSearch(opts);
    }

    return engine;
}

static SearchEngine *_parse_greedy(OptionParser &parser) {
    parser.document_synopsis("Greedy search (eager)", "");
    parser.document_note(
        "Open list",
        "In most cases, eager greedy best first search uses "
        "an alternation open list with one queue for each evaluator. "
        "If preferred operator heuristics are used, it adds an extra queue "
        "for each of these evaluators that includes only the nodes that "
        "are generated with a preferred operator. "
        "If only one evaluator and no preferred operator heuristic is used, "
        "the search does not use an alternation open list but a "
        "standard open list with only one queue.");
    parser.document_note(
        "Closed nodes",
        "Closed node are not re-opened");
    parser.document_note(
        "Equivalent statements using general eager search",
        "\n```\n--heuristic h2=eval2\n"
        "--search eager_greedy([eval1, h2], preferred=h2, boost=100)\n```\n"
        "is equivalent to\n"
        "```\n--heuristic h1=eval1 --heuristic h2=eval2\n"
        "--search eager(alt([single(h1), single(h1, pref_only=true), single(h2), \n"
        "                    single(h2, pref_only=true)], boost=100),\n"
        "               preferred=h2)\n```\n"
        "------------------------------------------------------------\n"
        "```\n--search eager_greedy([eval1, eval2])\n```\n"
        "is equivalent to\n"
        "```\n--search eager(alt([single(eval1), single(eval2)]))\n```\n"
        "------------------------------------------------------------\n"
        "```\n--heuristic h1=eval1\n"
        "--search eager_greedy(h1, preferred=h1)\n```\n"
        "is equivalent to\n"
        "```\n--heuristic h1=eval1\n"
        "--search eager(alt([single(h1), single(h1, pref_only=true)]),\n"
        "               preferred=h1)\n```\n"
        "------------------------------------------------------------\n"
        "```\n--search eager_greedy(eval1)\n```\n"
        "is equivalent to\n"
        "```\n--search eager(single(eval1))\n```\n", true);

    parser.add_list_option<ScalarEvaluator *>("evals", "scalar evaluators");
    parser.add_list_option<Heuristic *>(
        "preferred",
        "use preferred operators of these heuristics", "[]");
    parser.add_option<ap_float>(
        "boost",
        "boost value for preferred operator open lists", "0");

    add_pruning_option(parser);
    SearchEngine::add_options_to_parser(parser);

    Options opts = parser.parse();
    opts.verify_list_non_empty<ScalarEvaluator *>("evals");

    EagerSearch *engine = nullptr;
    if (!parser.dry_run()) {
        bool has_direct_h_key =
            opts.get_list<ScalarEvaluator *>("evals").size() == 1 &&
            opts.get_list<Heuristic *>("preferred").empty();
        opts.set("open", search_common::create_greedy_open_list_factory(opts));
        opts.set("reopen_closed", false);
        opts.set("mpd", false);
        opts.set("llm_h_open_list_key_index", has_direct_h_key ? 0 : -1);
        ScalarEvaluator *evaluator = nullptr;
        opts.set("f_eval", evaluator);
        engine = new EagerSearch(opts);
    }
    return engine;
}

static Plugin<SearchEngine> _plugin("eager", _parse);
static Plugin<SearchEngine> _plugin_astar("astar", _parse_astar);
static Plugin<SearchEngine> _plugin_greedy("eager_greedy", _parse_greedy);
}
