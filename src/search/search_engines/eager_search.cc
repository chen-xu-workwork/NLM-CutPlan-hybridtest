#include "eager_search.h"

#include "search_common.h"

#include "../evaluation_context.h"
#include "../globals.h"
#include "../heuristic.h"
#include "../option_parser.h"
#include "../plugin.h"
#include "../pruning_method.h"
#include "../successor_generator.h"
#include "../utils/timer.h"
#include "../utils/planvis.h"

#include "../open_lists/open_list_factory.h"

#include <algorithm>
#include <cassert>
#include <cerrno>
#include <cctype>
#include <condition_variable>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
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
static bool probes_enabled() {
    const char *value = getenv("NLM_EAGER_PROBES");
    if (!value)
        return false;

    string setting(value);
    return !setting.empty() && setting != "0" && setting != "false" &&
           setting != "FALSE";
}

static void probe_log(const string &message) {
    if (probes_enabled())
        cout << "[NLM-EAGER-PROBE] " << message << endl;
}

static int probe_int_env(const char *name, int default_value) {
    const char *value = getenv(name);
    if (!value)
        return default_value;

    int parsed = atoi(value);
    return parsed < 0 ? 0 : parsed;
}

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
    StateID state_id;
    size_t state_index;
    string state_label;
    string problem_id;
    string reason;
    ap_float g;
    ap_float h;
    string init;

    LLMRequest()
        : state_id(StateID::no_state), state_index(0), g(0), h(0) {
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
    if (!saw_status || !saw_actions) {
        result.error = "response is missing status or actions";
        return result;
    }
    skip_json_whitespace(body, position);
    if (position != body.size()) {
        result.error = "response contains trailing content";
        return result;
    }
    result.valid = true;
    return result;
}

static string normalize_operator_name(const string &raw_name) {
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
        body << "\"state_id\":" << request.state_index << ",";
        body << "\"state_label\":\"" << json_escape(request.state_label) << "\",";
        body << "\"problem_id\":\"" << json_escape(request.problem_id) << "\",";
        body << "\"reason\":\"" << json_escape(request.reason) << "\",";
        body << "\"g\":" << request.g << ",";
        body << "\"h\":" << request.h << ",";
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
             << " request_id=" << request.request_id
             << " state=" << request.state_label
             << " reason=" << request.reason
             << " g=" << request.g
             << " h=" << request.h
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
          stopping(false) {
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
};

static void probe_dump_pddl_init(const GlobalState &state,
                                 const SearchNode &node) {
    // 中文说明：这是调试探针，用于审阅展开状态导出的完整 (:init ...)。
    // 正式混合求解时不应对每个状态常驻开启，而应在判定需要 LLM 跳步
    // 的少量状态上按需调用 GlobalState::get_pddl_init_string()。
    if (!probes_enabled())
        return;

    static int dumped_states = 0;
    static int seen_expansions = 0;
    const int limit = probe_int_env("NLM_EAGER_STATE_PROBE_LIMIT", 3);
    const int stride = max(1, probe_int_env("NLM_EAGER_STATE_PROBE_STRIDE", 1));

    ++seen_expansions;
    if (limit == 0 || dumped_states >= limit)
        return;
    if ((seen_expansions - 1) % stride != 0)
        return;

    ++dumped_states;
    cout << "[NLM-EAGER-STATE-PROBE] begin"
         << " index=" << dumped_states
         << " state=" << state.get_id()
         << " g=" << node.get_g()
         << " real_g=" << node.get_real_g()
         << endl;
    cout << state.get_pddl_init_string();
    cout << "[NLM-EAGER-STATE-PROBE] end"
         << " index=" << dumped_states
         << " state=" << state.get_id()
         << endl;
}

class LLMTriggerMonitor {
    // 中文说明：这个 monitor 是 LLM 介入判定的轻量旁路层。
    // 它不改变 openlist 的内部结构，只在已有 h/g 计算点缓存信息，
    // 用于观察 plateau、父链停滞和全局 best-h 停滞。HTTP 模式会把
    // 候选交给 Python 主控；log 模式只记录触发，不改变搜索流程。
    struct Config {
        bool enabled;
        bool skip_pending;
        bool emit_state;
        bool request_initial;
        int frontier_k;
        int batch_size;
        int check_interval;
        int stall_expansions;
        int ancestor_depth;
        int min_depth;
        int max_pending;
        ap_float h_abs_epsilon;
        ap_float h_relative_epsilon;
        ap_float plateau_h_cv;
        ap_float plateau_f_cv;

        Config()
            : enabled(env_enabled("NLM_LLM_TRIGGER")),
              skip_pending(env_equals_ignore_case(
                  "NLM_LLM_PENDING_BEHAVIOR", "skip")),
              emit_state(env_enabled("NLM_LLM_EMIT_STATE")),
              request_initial(env_enabled("NLM_LLM_REQUEST_INITIAL")),
              frontier_k(max(1, env_int("NLM_LLM_FRONTIER_K", 64))),
              batch_size(max(1, env_int("NLM_LLM_BATCH_SIZE", 8))),
              check_interval(max(1, env_int("NLM_LLM_CHECK_INTERVAL", 50))),
              stall_expansions(env_int("NLM_LLM_STALL_EXPANSIONS", 500)),
              ancestor_depth(max(1, env_int("NLM_LLM_ANCESTOR_DEPTH", 4))),
              min_depth(max(0, env_int("NLM_LLM_MIN_DEPTH", 4))),
              max_pending(env_int("NLM_LLM_MAX_PENDING", 0)),
              h_abs_epsilon(env_float("NLM_LLM_H_EPSILON", 0.001)),
              h_relative_epsilon(max(
                  0.0, env_float("NLM_LLM_H_RELATIVE_EPSILON", 0.01))),
              plateau_h_cv(max(
                  0.0, env_float("NLM_LLM_PLATEAU_H_CV", 0.05))),
              plateau_f_cv(max(
                  0.0, env_float("NLM_LLM_PLATEAU_F_CV", 0.05))) {
        }
    };

    struct StateInfo {
        StateID state_id;
        StateID parent_id;
        ap_float g;
        ap_float h;
        int depth;

        StateInfo()
            : state_id(StateID::no_state), parent_id(StateID::no_state),
              g(0), h(0), depth(0) {
        }

        StateInfo(StateID state_id_, StateID parent_id_, ap_float g_,
                  ap_float h_, int depth_)
            : state_id(state_id_), parent_id(parent_id_), g(g_), h(h_),
              depth(depth_) {
        }
    };

    struct FrontierEntry {
        ap_float f;
        ap_float h;
        ap_float g;
        StateID state_id;
        int insert_id;

        FrontierEntry(ap_float f_, ap_float h_, ap_float g_,
                      StateID state_id_, int insert_id_)
            : f(f_), h(h_), g(g_), state_id(state_id_),
              insert_id(insert_id_) {
        }
    };

    struct FrontierEntryLess {
        bool operator()(const FrontierEntry &lhs,
                        const FrontierEntry &rhs) const {
            if (lhs.f != rhs.f)
                return lhs.f < rhs.f;
            if (lhs.h != rhs.h)
                return lhs.h < rhs.h;
            if (lhs.g != rhs.g)
                return lhs.g < rhs.g;
            if (lhs.insert_id != rhs.insert_id)
                return lhs.insert_id < rhs.insert_id;
            return lhs.state_id.hash() < rhs.state_id.hash();
        }
    };

    Config config;
    unordered_map<StateID, StateInfo> state_infos;
    unordered_set<StateID> pending_states;
    unordered_set<StateID> suspended_states;
    unordered_set<StateID> requested_states;
    multiset<FrontierEntry, FrontierEntryLess> frontier;
    LLMBridge bridge;
    int next_insert_id;
    int next_request_id;
    int expansions;
    int expansions_since_best_h;
    int last_frontier_check_expansion;
    ap_float best_h;

    bool reached_pending_limit() const {
        return config.max_pending > 0 &&
               static_cast<int>(pending_states.size()) >= config.max_pending;
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
        request_id << state_id.hash() << "-" << next_request_id++;
        request.request_id = request_id.str();
        request.state_id = state_id;
        request.state_index = state_id.hash();
        request.state_label = state_id_label(state_id);
        request.problem_id = env_string(
            "NLM_LLM_PROBLEM_ID", env_string("NLM_PROBLEM_ID", ""));
        request.reason = reason;
        request.g = g;
        request.h = h;

        GlobalState state = g_state_registry->lookup_state(state_id);
        request.init = state.get_pddl_init_string();
        return request;
    }

    bool request_state(StateID state_id, const string &reason,
                       ap_float g, ap_float h) {
        if (!config.enabled || state_id == StateID::no_state)
            return false;
        if (requested_states.count(state_id))
            return false;
        if (reached_pending_limit())
            return false;

        LLMRequest request = make_request(state_id, reason, g, h);
        if (!bridge.submit(request))
            return false;

        requested_states.insert(state_id);
        // Log mode is observational and never produces a completion. Treating
        // those requests as pending would make an empty Open List wait forever.
        if (bridge.expects_response())
            pending_states.insert(state_id);
        return true;
    }

    bool is_valid_frontier_entry(const FrontierEntry &entry,
                                  SearchSpace &search_space) const {
        GlobalState state = g_state_registry->lookup_state(entry.state_id);
        SearchNode node = search_space.get_node(state);
        if (!node.is_open())
            return false;
        if (std::fabs(node.get_g() - entry.g) > config.h_abs_epsilon)
            return false;
        return true;
    }

    vector<FrontierEntry> collect_top_frontier(SearchSpace &search_space) {
        vector<FrontierEntry> result;
        auto it = frontier.begin();
        while (it != frontier.end() &&
               static_cast<int>(result.size()) < config.frontier_k) {
            if (!is_valid_frontier_entry(*it, search_space)) {
                auto stale_it = it++;
                frontier.erase(stale_it);
                continue;
            }
            if (!pending_states.count(it->state_id) &&
                !requested_states.count(it->state_id)) {
                result.push_back(*it);
            }
            ++it;
        }
        return result;
    }

    bool low_coefficient_of_variation(
        const vector<FrontierEntry> &entries, bool use_h) const {
        if (entries.size() < 2)
            return false;

        ap_float sum = 0;
        for (const FrontierEntry &entry : entries)
            sum += use_h ? entry.h : entry.f;
        ap_float mean = sum / entries.size();

        ap_float squared_diff_sum = 0;
        for (const FrontierEntry &entry : entries) {
            ap_float value = use_h ? entry.h : entry.f;
            ap_float diff = value - mean;
            squared_diff_sum += diff * diff;
        }
        ap_float stddev = sqrt(squared_diff_sum / entries.size());

        if (std::fabs(mean) <= config.h_abs_epsilon)
            return stddev <= config.h_abs_epsilon;

        ap_float cv = stddev / std::fabs(mean);
        return cv <= (use_h ? config.plateau_h_cv : config.plateau_f_cv);
    }

    bool frontier_plateau(const vector<FrontierEntry> &entries) const {
        if (entries.size() < 2)
            return false;
        return low_coefficient_of_variation(entries, true) &&
               low_coefficient_of_variation(entries, false);
    }

    bool ancestor_stagnant(StateID state_id) const {
        auto state_it = state_infos.find(state_id);
        if (state_it == state_infos.end())
            return false;
        const StateInfo &info = state_it->second;
        if (info.depth < config.min_depth)
            return false;

        StateID parent_id = info.parent_id;
        ap_float best_improvement = 0;
        int inspected = 0;
        while (parent_id != StateID::no_state &&
               inspected < config.ancestor_depth) {
            auto parent_it = state_infos.find(parent_id);
            if (parent_it == state_infos.end())
                break;
            const StateInfo &parent_info = parent_it->second;
            best_improvement =
                max(best_improvement, parent_info.h - info.h);
            parent_id = parent_info.parent_id;
            ++inspected;
        }

        return inspected == config.ancestor_depth &&
               best_improvement <=
               h_improvement_threshold(info.h + best_improvement, info.h);
    }

public:
    LLMTriggerMonitor()
        : next_insert_id(0),
          next_request_id(0),
          expansions(0),
          expansions_since_best_h(0),
          last_frontier_check_expansion(0),
          best_h(numeric_limits<ap_float>::infinity()) {
        if (config.enabled) {
            cout << "[NLM-LLM-TRIGGER] enabled"
                 << " frontier_k=" << config.frontier_k
                 << " batch_size=" << config.batch_size
                 << " check_interval=" << config.check_interval
                 << " stall_expansions=" << config.stall_expansions
                 << " ancestor_depth=" << config.ancestor_depth
                 << " min_depth=" << config.min_depth
                 << " max_pending=" << config.max_pending
                 << " h_abs_epsilon=" << config.h_abs_epsilon
                 << " h_relative_epsilon=" << config.h_relative_epsilon
                 << " plateau_h_cv=" << config.plateau_h_cv
                 << " plateau_f_cv=" << config.plateau_f_cv
                 << " emit_state=" << (config.emit_state ? 1 : 0)
                 << " request_initial="
                 << (config.request_initial ? 1 : 0)
                 << " pending_behavior="
                 << (config.skip_pending ? "skip" : "normal")
                 << endl;
        }
    }

    ~LLMTriggerMonitor() {
        bridge.stop();
        poll_bridge();
    }

    bool enabled() const {
        return config.enabled;
    }

    void start_bridge() {
        if (config.enabled)
            bridge.start();
    }

    vector<LLMResponse> poll_bridge() {
        if (!config.enabled)
            return vector<LLMResponse>();
        vector<LLMResponse> responses = bridge.poll_completed();
        for (const LLMResponse &response : responses) {
            pending_states.erase(response.state_id);
            cout << "[NLM-LLM-BRIDGE] completed"
                 << " request_id=" << response.request_id
                 << " state=" << response.state_label
                 << " transport_ok=" << (response.transport_ok ? 1 : 0)
                 << " http_status=" << response.http_status
                 << " body_bytes=" << response.body.size();
            if (!response.error.empty())
                cout << " error=\"" << response.error << "\"";
            cout << endl;
            if (!response.body.empty()) {
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
        return request_state(state_id, "initial_replay_test", g, h);
    }

    void record_open_state(StateID state_id, StateID parent_id,
                           ap_float g, ap_float h) {
        if (!config.enabled)
            return;
        if (!std::isfinite(static_cast<double>(h)))
            return;

        int depth = 0;
        if (parent_id != StateID::no_state) {
            auto parent_it = state_infos.find(parent_id);
            if (parent_it != state_infos.end())
                depth = parent_it->second.depth + 1;
        }

        state_infos[state_id] = StateInfo(state_id, parent_id, g, h, depth);
        frontier.insert(FrontierEntry(g + h, h, g, state_id,
                                      next_insert_id++));

        if (meaningfully_improves_h(best_h, h)) {
            best_h = h;
            expansions_since_best_h = 0;
        }
    }

    void record_frontier_reinsert(StateID state_id, ap_float g, ap_float h) {
        if (!config.enabled)
            return;
        if (!std::isfinite(static_cast<double>(h)))
            return;

        auto info_it = state_infos.find(state_id);
        if (info_it != state_infos.end()) {
            info_it->second.g = g;
            info_it->second.h = h;
        }
        frontier.insert(FrontierEntry(g + h, h, g, state_id,
                                      next_insert_id++));

        if (meaningfully_improves_h(best_h, h)) {
            best_h = h;
            expansions_since_best_h = 0;
        }
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

    bool consider_popped_state(StateID state_id) {
        if (!config.enabled)
            return false;
        if (pending_states.count(state_id))
            return false;
        if (!ancestor_stagnant(state_id))
            return false;

        const StateInfo &info = state_infos.find(state_id)->second;
        return request_state(state_id, "ancestor_stagnation",
                             info.g, info.h);
    }

    void record_expanded(StateID /*state_id*/) {
        if (!config.enabled)
            return;
        ++expansions;
        ++expansions_since_best_h;
    }

    void maybe_emit_frontier_batch(SearchSpace &search_space) {
        if (!config.enabled)
            return;
        if (expansions - last_frontier_check_expansion <
            config.check_interval) {
            return;
        }
        last_frontier_check_expansion = expansions;

        vector<FrontierEntry> top_entries = collect_top_frontier(search_space);
        bool plateau = frontier_plateau(top_entries);
        bool global_stall = config.stall_expansions > 0 &&
            expansions_since_best_h >= config.stall_expansions;
        if (!plateau && !global_stall)
            return;

        string reason;
        if (plateau && global_stall)
            reason = "frontier_plateau+global_stall";
        else if (plateau)
            reason = "frontier_plateau";
        else
            reason = "global_stall";

        int emitted = 0;
        for (const FrontierEntry &entry : top_entries) {
            if (request_state(entry.state_id, reason, entry.g, entry.h)) {
                ++emitted;
                if (emitted >= config.batch_size)
                    break;
            }
        }
    }
};

EagerSearch::EagerSearch(const Options &opts)
    : SearchEngine(opts),
      reopen_closed_nodes(opts.get<bool>("reopen_closed")),
      use_multi_path_dependence(opts.get<bool>("mpd")),
      open_list(opts.get<shared_ptr<OpenListFactory>>("open")->
                create_state_open_list()),
      f_evaluator(opts.get<ScalarEvaluator *>("f_eval", nullptr)),
      preferred_operator_heuristics(opts.get_list<Heuristic *>("preferred")),
      pruning_method(opts.get<shared_ptr<PruningMethod>>("pruning")),
      llm_trigger_monitor(new LLMTriggerMonitor()) {
}

EagerSearch::~EagerSearch() = default;

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

    if (llm_trigger_monitor->enabled()) {
        llm_operator_by_name.reserve(g_operators.size());
        for (const GlobalOperator &op : g_operators) {
            llm_operator_by_name[
                normalize_operator_name(op.get_name())] = &op;
        }
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
                eval_context.get_heuristic_value(heuristics[0]);
            llm_trigger_monitor->record_open_state(
                initial_state.get_id(), StateID::no_state, 0, initial_h);
            llm_trigger_monitor->maybe_request_initial(
                initial_state.get_id(), 0, initial_h);
        }
        if (probes_enabled()) {
            ostringstream message;
            message << "openlist_insert initial state=" << initial_state.get_id()
                    << " g=0";
            probe_log(message.str());
        }
    }

    print_initial_h_values(eval_context);
    if (PLAN_VIS_LOG == plan_vis_log) {
        utils::Timer h_time;
        if (PLAN_VIS_LOG) h_time.reset();
        ap_float h_val = eval_context.get_heuristic_value(heuristics[0]);
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
    ap_float source_h = eval_context.get_heuristic_value(heuristics[0]);
    llm_trigger_monitor->record_frontier_reinsert(
        source_id, source_node.get_g(), source_h);
    cout << "[NLM-LLM-INJECT] requeued source="
         << state_id_label(source_id)
         << " reason=resume_after_llm"
         << endl;
}

bool EagerSearch::inject_llm_action_chain(
    StateID source_id, const vector<string> &actions) {
    GlobalState current_state = g_state_registry->lookup_state(source_id);
    int applied_actions = 0;
    int inserted_states = 0;

    for (const string &raw_action : actions) {
        string requested_name = normalize_operator_name(raw_action);
        const GlobalOperator *selected_operator = nullptr;
        auto operator_it = llm_operator_by_name.find(requested_name);
        if (operator_it != llm_operator_by_name.end())
            selected_operator = operator_it->second;
        if (!selected_operator) {
            cout << "[NLM-LLM-INJECT] unknown action=\"" << raw_action
                 << "\" source=" << state_id_label(source_id) << endl;
            break;
        }
        if (!selected_operator->is_applicable(current_state)) {
            cout << "[NLM-LLM-INJECT] action no longer applicable=\""
                 << raw_action << "\" state=" << current_state.get_id()
                 << endl;
            break;
        }

        SearchNode parent_node = search_space.get_node(current_state);
        if (parent_node.is_dead_end() ||
            parent_node.get_real_g() + selected_operator->get_cost() >= bound) {
            break;
        }

        GlobalState successor_state =
            g_state_registry->get_successor_state(
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
                eval_context.get_heuristic_value(heuristics[0]);
            successor_node.open(parent_node, selected_operator);
            open_list->insert(eval_context, successor_state.get_id());
            llm_trigger_monitor->record_open_state(
                successor_state.get_id(), current_state.get_id(),
                successor_g, successor_h);
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
                    eval_context.get_heuristic_value(heuristics[0]);
                llm_trigger_monitor->record_open_state(
                    successor_state.get_id(), current_state.get_id(),
                    successor_node.get_g(), successor_h);
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

    cout << "[NLM-LLM-INJECT] chain source=" << state_id_label(source_id)
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
                cout << "[NLM-LLM-INJECT] response"
                     << " request_id=" << response.request_id
                     << " state=" << response.state_label
                     << " status=" << parsed.status
                     << " actions=" << parsed.actions.size()
                     << endl;
                bool usable_status =
                    parsed.status == "ok" || parsed.status == "partial";
                if (usable_status && !parsed.actions.empty()) {
                    inject_llm_action_chain(
                        response.state_id, parsed.actions);
                } else if (!parsed.actions.empty()) {
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
    size_t applicable_ops_before_pruning = applicable_ops.size();

    /*
      TODO: When preferred operators are in use, a preferred operator will be
      considered by the preferred operator queues even when it is pruned.
    */
    pruning_method->prune_operators(s, applicable_ops);
    if (probes_enabled()) {
        ostringstream message;
        message << "successor_generation parent=" << s.get_id()
                << " generated=" << applicable_ops_before_pruning
                << " after_pruning=" << applicable_ops.size();
        probe_log(message.str());
    }

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
        if (probes_enabled()) {
            ostringstream message;
            message << "generate_successor parent=" << s.get_id()
                    << " op=\"" << op->get_name() << "\""
                    << " succ=" << succ_state.get_id()
                    << " op_cost=" << op->get_cost();
            probe_log(message.str());
        }
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
                if (probes_enabled()) {
                    ostringstream message;
                    message << "heuristic_dead_end succ=" << succ_state.get_id()
                            << " parent=" << s.get_id()
                            << " op=\"" << op->get_name() << "\""
                            << " g=" << succ_g;
                    probe_log(message.str());
                }
                continue;
            }
            utils::Timer h_time;
            if (PLAN_VIS_LOG == plan_vis_log) {h_time.reset();
            assert(false); // todo: code changed here, have to fetch time somewhere else
			}
//            ap_float succ_h = 4;
//  was before:
            ap_float succ_h = eval_context.get_heuristic_value(heuristics[0]);
            if (probes_enabled()) {
                ostringstream message;
                message << "heuristic_eval succ=" << succ_state.get_id()
                        << " parent=" << s.get_id()
                        << " op=\"" << op->get_name() << "\""
                        << " g=" << succ_g
                        << " h=" << succ_h
                        << " preferred=" << (is_preferred ? 1 : 0);
                probe_log(message.str());
            }
            if (PLAN_VIS_LOG == plan_vis_log) h_time.stop();

            succ_node.open(node, op);

            open_list->insert(eval_context, succ_state.get_id());
            llm_trigger_monitor->record_open_state(
                succ_state.get_id(), s.get_id(), succ_g, succ_h);
            if (probes_enabled()) {
                ostringstream message;
                message << "openlist_insert state=" << succ_state.get_id()
                        << " parent=" << s.get_id()
                        << " op=\"" << op->get_name() << "\""
                        << " g=" << succ_g
                        << " h=" << succ_h
                        << " preferred=" << (is_preferred ? 1 : 0);
                probe_log(message.str());
            }
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
                        eval_context.get_heuristic_value(heuristics[0]);
                    llm_trigger_monitor->record_open_state(
                        succ_state.get_id(), s.get_id(),
                        succ_node.get_g(), reopen_h);
                }
                if (probes_enabled()) {
                    ostringstream message;
                    message << "openlist_reinsert reopened state="
                            << succ_state.get_id()
                            << " parent=" << s.get_id()
                            << " op=\"" << op->get_name() << "\""
                            << " g=" << succ_node.get_g()
                            << " preferred=" << (is_preferred ? 1 : 0);
                    probe_log(message.str());
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
        llm_trigger_monitor->maybe_emit_frontier_batch(search_space);
        vector<ap_float> last_key_removed;
        StateID id = open_list->remove_min(
            use_multi_path_dependence ? &last_key_removed : nullptr);
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
            assert(last_key_removed.size() == 2);
            if (node.is_dead_end())
                continue;
            ap_float pushed_h = last_key_removed[1];

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
                            eval_context.get_result(heuristics[0]).get_h_value();
                        llm_trigger_monitor->record_frontier_reinsert(
                            node.get_state_id(), node.get_g(), current_h);
                    }
                    if (probes_enabled()) {
                        ostringstream message;
                        message << "openlist_reinsert stale_mpd state="
                                << node.get_state_id()
                                << " g=" << node.get_g()
                                << " pushed_h=" << pushed_h
                                << " current_h="
                                << eval_context.get_result(heuristics[0]).get_h_value();
                        probe_log(message.str());
                    }
                    continue;
                }
            }
        }

        bool requested_for_llm =
            llm_trigger_monitor->consider_popped_state(id);
        if (requested_for_llm &&
            llm_trigger_monitor->suspend_if_pending(id)) {
            continue;
        }

        node.close();
        assert(!node.is_dead_end());
        update_f_value_statistics(node);
        statistics.inc_expanded();
        llm_trigger_monitor->record_expanded(id);
        if (probes_enabled()) {
            ostringstream message;
            message << "expand state=" << s.get_id()
                    << " g=" << node.get_g()
                    << " real_g=" << node.get_real_g();
            probe_log(message.str());
        }
        probe_dump_pddl_init(s, node);
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
        opts.set("open", search_common::create_greedy_open_list_factory(opts));
        opts.set("reopen_closed", false);
        opts.set("mpd", false);
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
