#include "trajectory_score_protocol.h"

#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>

using namespace std;

namespace trajectory_score_protocol {
namespace {
enum class JsonType {
    NULL_VALUE,
    BOOL_VALUE,
    NUMBER_VALUE,
    STRING_VALUE,
    ARRAY_VALUE,
    OBJECT_VALUE
};

struct JsonValue {
    JsonType type;
    bool bool_value;
    double number_value;
    string string_value;
    vector<JsonValue> array_value;
    map<string, JsonValue> object_value;

    JsonValue()
        : type(JsonType::NULL_VALUE), bool_value(false), number_value(0.0) {
    }

    static JsonValue boolean(bool value) {
        JsonValue result;
        result.type = JsonType::BOOL_VALUE;
        result.bool_value = value;
        return result;
    }

    static JsonValue number(double value) {
        JsonValue result;
        result.type = JsonType::NUMBER_VALUE;
        result.number_value = value;
        return result;
    }

    static JsonValue string_value_of(const string &value) {
        JsonValue result;
        result.type = JsonType::STRING_VALUE;
        result.string_value = value;
        return result;
    }

    static JsonValue array(const vector<JsonValue> &value) {
        JsonValue result;
        result.type = JsonType::ARRAY_VALUE;
        result.array_value = value;
        return result;
    }

    static JsonValue object(const map<string, JsonValue> &value) {
        JsonValue result;
        result.type = JsonType::OBJECT_VALUE;
        result.object_value = value;
        return result;
    }
};

class JsonParser {
    const string &text;
    size_t position;
    size_t depth;

    void skip_whitespace() {
        while (position < text.size()) {
            char ch = text[position];
            if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n')
                ++position;
            else
                break;
        }
    }

    void fail(const string &message) const {
        ostringstream out;
        out << message << " at byte " << position;
        throw runtime_error(out.str());
    }

    static int hex_value(char ch) {
        if (ch >= '0' && ch <= '9')
            return ch - '0';
        if (ch >= 'a' && ch <= 'f')
            return ch - 'a' + 10;
        if (ch >= 'A' && ch <= 'F')
            return ch - 'A' + 10;
        return -1;
    }

    unsigned int parse_hex_quad() {
        if (position + 4 > text.size())
            fail("truncated unicode escape");
        unsigned int value = 0;
        for (int index = 0; index < 4; ++index) {
            int digit = hex_value(text[position++]);
            if (digit < 0)
                fail("invalid unicode escape");
            value = value * 16 + static_cast<unsigned int>(digit);
        }
        return value;
    }

    static void append_utf8(string &output, unsigned int codepoint) {
        if (codepoint <= 0x7f) {
            output += static_cast<char>(codepoint);
        } else if (codepoint <= 0x7ff) {
            output += static_cast<char>(0xc0 | (codepoint >> 6));
            output += static_cast<char>(0x80 | (codepoint & 0x3f));
        } else if (codepoint <= 0xffff) {
            output += static_cast<char>(0xe0 | (codepoint >> 12));
            output += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
            output += static_cast<char>(0x80 | (codepoint & 0x3f));
        } else {
            output += static_cast<char>(0xf0 | (codepoint >> 18));
            output += static_cast<char>(0x80 | ((codepoint >> 12) & 0x3f));
            output += static_cast<char>(0x80 | ((codepoint >> 6) & 0x3f));
            output += static_cast<char>(0x80 | (codepoint & 0x3f));
        }
    }

    string parse_string() {
        if (position >= text.size() || text[position] != '"')
            fail("expected string");
        ++position;
        string output;
        while (position < text.size()) {
            unsigned char ch = static_cast<unsigned char>(text[position++]);
            if (ch == '"')
                return output;
            if (ch < 0x20)
                fail("unescaped control character in string");
            if (ch != '\\') {
                output += static_cast<char>(ch);
                continue;
            }
            if (position >= text.size())
                fail("truncated string escape");
            char escape = text[position++];
            switch (escape) {
            case '"': output += '"'; break;
            case '\\': output += '\\'; break;
            case '/': output += '/'; break;
            case 'b': output += '\b'; break;
            case 'f': output += '\f'; break;
            case 'n': output += '\n'; break;
            case 'r': output += '\r'; break;
            case 't': output += '\t'; break;
            case 'u': {
                unsigned int codepoint = parse_hex_quad();
                if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                    if (position + 2 > text.size() ||
                        text[position] != '\\' || text[position + 1] != 'u')
                        fail("missing low unicode surrogate");
                    position += 2;
                    unsigned int low = parse_hex_quad();
                    if (low < 0xdc00 || low > 0xdfff)
                        fail("invalid low unicode surrogate");
                    codepoint = 0x10000 +
                        ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                    fail("unexpected low unicode surrogate");
                }
                append_utf8(output, codepoint);
                break;
            }
            default:
                fail("invalid string escape");
            }
        }
        fail("unterminated string");
        return string();
    }

    JsonValue parse_number() {
        size_t begin = position;
        if (text[position] == '-')
            ++position;
        if (position >= text.size())
            fail("truncated number");
        if (text[position] == '0') {
            ++position;
            if (position < text.size() && text[position] >= '0' &&
                text[position] <= '9')
                fail("leading zero in number");
        } else {
            if (text[position] < '1' || text[position] > '9')
                fail("invalid number");
            while (position < text.size() && text[position] >= '0' &&
                   text[position] <= '9')
                ++position;
        }
        if (position < text.size() && text[position] == '.') {
            ++position;
            if (position >= text.size() || text[position] < '0' ||
                text[position] > '9')
                fail("invalid fraction");
            while (position < text.size() && text[position] >= '0' &&
                   text[position] <= '9')
                ++position;
        }
        if (position < text.size() &&
            (text[position] == 'e' || text[position] == 'E')) {
            ++position;
            if (position < text.size() &&
                (text[position] == '+' || text[position] == '-'))
                ++position;
            if (position >= text.size() || text[position] < '0' ||
                text[position] > '9')
                fail("invalid exponent");
            while (position < text.size() && text[position] >= '0' &&
                   text[position] <= '9')
                ++position;
        }

        string token = text.substr(begin, position - begin);
        errno = 0;
        char *end = nullptr;
        double value = strtod(token.c_str(), &end);
        if (errno == ERANGE || end != token.c_str() + token.size() ||
            !isfinite(value))
            fail("number is out of range");
        return JsonValue::number(value);
    }

    JsonValue parse_array() {
        ++position;
        ++depth;
        if (depth > 64)
            fail("JSON nesting limit exceeded");
        vector<JsonValue> values;
        skip_whitespace();
        if (position < text.size() && text[position] == ']') {
            ++position;
            --depth;
            return JsonValue::array(values);
        }
        while (true) {
            values.push_back(parse_value());
            skip_whitespace();
            if (position >= text.size())
                fail("unterminated array");
            char delimiter = text[position++];
            if (delimiter == ']')
                break;
            if (delimiter != ',')
                fail("expected ',' or ']' in array");
            skip_whitespace();
        }
        --depth;
        return JsonValue::array(values);
    }

    JsonValue parse_object() {
        ++position;
        ++depth;
        if (depth > 64)
            fail("JSON nesting limit exceeded");
        map<string, JsonValue> values;
        skip_whitespace();
        if (position < text.size() && text[position] == '}') {
            ++position;
            --depth;
            return JsonValue::object(values);
        }
        while (true) {
            skip_whitespace();
            string key = parse_string();
            skip_whitespace();
            if (position >= text.size() || text[position] != ':')
                fail("expected ':' in object");
            ++position;
            skip_whitespace();
            JsonValue value = parse_value();
            if (!values.emplace(key, value).second)
                fail("duplicate object key '" + key + "'");
            skip_whitespace();
            if (position >= text.size())
                fail("unterminated object");
            char delimiter = text[position++];
            if (delimiter == '}')
                break;
            if (delimiter != ',')
                fail("expected ',' or '}' in object");
            skip_whitespace();
        }
        --depth;
        return JsonValue::object(values);
    }

    JsonValue parse_value() {
        skip_whitespace();
        if (position >= text.size())
            fail("expected JSON value");
        char ch = text[position];
        if (ch == '"')
            return JsonValue::string_value_of(parse_string());
        if (ch == '{')
            return parse_object();
        if (ch == '[')
            return parse_array();
        if (ch == '-' || (ch >= '0' && ch <= '9'))
            return parse_number();
        if (text.compare(position, 4, "true") == 0) {
            position += 4;
            return JsonValue::boolean(true);
        }
        if (text.compare(position, 5, "false") == 0) {
            position += 5;
            return JsonValue::boolean(false);
        }
        if (text.compare(position, 4, "null") == 0) {
            position += 4;
            return JsonValue();
        }
        fail("invalid JSON value");
        return JsonValue();
    }

public:
    explicit JsonParser(const string &text_)
        : text(text_), position(0), depth(0) {
    }

    JsonValue parse() {
        JsonValue result = parse_value();
        skip_whitespace();
        if (position != text.size())
            fail("trailing content after JSON value");
        return result;
    }
};

const JsonValue *find_field(const JsonValue &object, const string &key) {
    auto it = object.object_value.find(key);
    if (it == object.object_value.end())
        return nullptr;
    return &it->second;
}

bool read_string_field(
    const JsonValue &object, const string &key, string &output,
    bool required, string &error) {
    const JsonValue *value = find_field(object, key);
    if (!value) {
        if (required)
            error = "missing required field '" + key + "'";
        return !required;
    }
    if (value->type != JsonType::STRING_VALUE) {
        error = "field '" + key + "' must be a string";
        return false;
    }
    output = value->string_value;
    return true;
}

bool read_int_field(
    const JsonValue &object, const string &key, int &output,
    bool required, string &error) {
    const JsonValue *value = find_field(object, key);
    if (!value) {
        if (required)
            error = "missing required field '" + key + "'";
        return !required;
    }
    if (value->type != JsonType::NUMBER_VALUE ||
        floor(value->number_value) != value->number_value ||
        value->number_value < numeric_limits<int>::min() ||
        value->number_value > numeric_limits<int>::max()) {
        error = "field '" + key + "' must be an integer";
        return false;
    }
    output = static_cast<int>(value->number_value);
    return true;
}

void append_json_string(ostringstream &out, const string &value) {
    static const char *HEX = "0123456789abcdef";
    out << '"';
    for (unsigned char ch : value) {
        switch (ch) {
        case '"': out << "\\\""; break;
        case '\\': out << "\\\\"; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                out << "\\u00" << HEX[(ch >> 4) & 0xf] << HEX[ch & 0xf];
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    out << '"';
}

void append_optional_string(
    ostringstream &out, bool has_value, const string &value) {
    if (has_value)
        append_json_string(out, value);
    else
        out << "null";
}

void append_optional_number(
    ostringstream &out, bool has_value, double value) {
    if (has_value && isfinite(value))
        out << setprecision(17) << value;
    else
        out << "null";
}

void append_state(ostringstream &out, const TrajectoryStateInfo &state) {
    out << "{\"state_index\":" << state.state_index
        << ",\"state_id\":" << state.state_id
        << ",\"h\":";
    append_optional_number(out, state.has_h, state.h);
    out << '}';
}
}

ScoreRequest::ScoreRequest()
    : protocol_version(PROTOCOL_VERSION) {
}

ParsedMessage::ParsedMessage()
    : kind(MessageKind::INVALID) {
}

ParsedMessage parse_message(const string &json_line) {
    ParsedMessage parsed;
    try {
        JsonValue root = JsonParser(json_line).parse();
        if (root.type != JsonType::OBJECT_VALUE) {
            parsed.error = "top-level JSON value must be an object";
            return parsed;
        }

        string type;
        if (!read_string_field(root, "type", type, true, parsed.error))
            return parsed;
        read_string_field(
            root, "request_id", parsed.request.request_id, false,
            parsed.error);
        read_string_field(
            root, "problem_id", parsed.request.problem_id, false,
            parsed.error);
        read_string_field(
            root, "task_hash", parsed.request.task_hash, false,
            parsed.error);
        if (!parsed.error.empty())
            return parsed;

        if (type == "shutdown") {
            int version = PROTOCOL_VERSION;
            if (!read_int_field(
                    root, "protocol_version", version, false, parsed.error))
                return parsed;
            if (version != PROTOCOL_VERSION) {
                parsed.error = "unsupported protocol_version";
                return parsed;
            }
            parsed.request.protocol_version = version;
            parsed.kind = MessageKind::SHUTDOWN;
            return parsed;
        }
        if (type != "score_request") {
            parsed.error = "field 'type' must be 'score_request' or 'shutdown'";
            return parsed;
        }

        if (!read_int_field(
                root, "protocol_version", parsed.request.protocol_version,
                true, parsed.error))
            return parsed;
        if (parsed.request.protocol_version != PROTOCOL_VERSION) {
            parsed.error = "unsupported protocol_version";
            return parsed;
        }
        if (!read_string_field(
                root, "request_id", parsed.request.request_id, true,
                parsed.error) ||
            parsed.request.request_id.empty()) {
            if (parsed.error.empty())
                parsed.error = "field 'request_id' must not be empty";
            return parsed;
        }
        if (!read_string_field(
                root, "problem_id", parsed.request.problem_id, true,
                parsed.error) ||
            parsed.request.problem_id.empty()) {
            if (parsed.error.empty())
                parsed.error = "field 'problem_id' must not be empty";
            return parsed;
        }
        if (!read_string_field(
                root, "task_hash", parsed.request.task_hash, true,
                parsed.error) ||
            parsed.request.task_hash.empty()) {
            if (parsed.error.empty())
                parsed.error = "field 'task_hash' must not be empty";
            return parsed;
        }

        const JsonValue *actions = find_field(root, "actions");
        if (!actions) {
            parsed.error = "missing required field 'actions'";
            return parsed;
        }
        if (actions->type != JsonType::ARRAY_VALUE) {
            parsed.error = "field 'actions' must be an array";
            return parsed;
        }
        for (const JsonValue &action : actions->array_value) {
            if (action.type != JsonType::STRING_VALUE) {
                parsed.error = "every item in 'actions' must be a string";
                return parsed;
            }
            parsed.request.actions.push_back(action.string_value);
        }

        parsed.kind = MessageKind::SCORE_REQUEST;
        return parsed;
    } catch (const exception &error) {
        parsed.error = error.what();
        return parsed;
    }
}

string serialize_ready(
    const string &task_hash, const string &heuristic_config,
    size_t max_request_bytes) {
    ostringstream out;
    out << "{\"type\":\"ready\",\"protocol_version\":"
        << PROTOCOL_VERSION << ",\"task_hash\":";
    append_json_string(out, task_hash);
    out << ",\"heuristic_config\":";
    append_json_string(out, heuristic_config);
    out << ",\"max_request_bytes\":" << max_request_bytes << '}';
    return out.str();
}

string serialize_score_response(
    const ScoreRequest &request, const string &loaded_task_hash,
    const ActionChainEvaluationResult &result,
    bool recycle_recommended, const string &recycle_reason) {
    ostringstream out;
    out << "{\"type\":\"score_response\",\"protocol_version\":"
        << PROTOCOL_VERSION << ",\"request_id\":";
    append_json_string(out, request.request_id);
    out << ",\"problem_id\":";
    append_json_string(out, request.problem_id);
    out << ",\"task_hash\":";
    append_json_string(out, loaded_task_hash);
    out << ",\"status\":";
    append_json_string(out, result.status);
    out << ",\"outcome\":";
    append_optional_string(out, !result.outcome.empty(), result.outcome);
    out << ",\"generated_action_count\":"
        << result.generated_action_count;
    out << ",\"applied_action_count\":"
        << result.applied_action_count;
    out << ",\"invalid_action_index\":";
    if (result.has_invalid_action)
        out << result.invalid_action_index;
    else
        out << "null";
    out << ",\"path_cost\":" << setprecision(17) << result.path_cost;
    out << ",\"registered_state_count\":"
        << result.registered_state_count;
    out << ",\"scorer_seconds\":"
        << setprecision(17) << result.scorer_seconds;
    out << ",\"recycle_recommended\":"
        << (recycle_recommended ? "true" : "false");
    out << ",\"recycle_reason\":";
    append_optional_string(out, recycle_recommended, recycle_reason);
    out << ",\"error\":";
    append_optional_string(
        out, !result.error_message.empty(), result.error_message);
    out << ",\"states\":[";
    for (size_t index = 0; index < result.states.size(); ++index) {
        if (index)
            out << ',';
        append_state(out, result.states[index]);
    }
    out << "]}";
    return out.str();
}

string serialize_error_response(
    const ScoreRequest &request, const string &loaded_task_hash,
    const string &status, const string &error_message) {
    ActionChainEvaluationResult result;
    result.status = status;
    result.error_message = error_message;
    result.generated_action_count = request.actions.size();
    result.registered_state_count = 0;
    return serialize_score_response(
        request, loaded_task_hash, result, false, string());
}

string serialize_shutdown_ack(
    const string &request_id, const string &task_hash) {
    ostringstream out;
    out << "{\"type\":\"shutdown_ack\",\"protocol_version\":"
        << PROTOCOL_VERSION << ",\"request_id\":";
    append_json_string(out, request_id);
    out << ",\"task_hash\":";
    append_json_string(out, task_hash);
    out << '}';
    return out.str();
}

}
