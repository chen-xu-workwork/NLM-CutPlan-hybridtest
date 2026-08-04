#include "trajectory_scorer_main.h"

#include "action_chain_evaluator.h"
#include "evaluation_context.h"
#include "globals.h"
#include "heuristic.h"
#include "option_parser.h"
#include "state_registry.h"
#include "trajectory_score_protocol.h"
#include "utils/system.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace std;
using trajectory_score_protocol::MessageKind;
using trajectory_score_protocol::ParsedMessage;
using trajectory_score_protocol::ScoreRequest;

namespace trajectory_scorer {
namespace {
const char *DEFAULT_HEURISTIC =
    "lmcutnumeric(use_second_order_simple=true,bound_iterations=10,"
    "ceiling_less_than_one=true)";

enum class RunMode {
    NONE,
    ONCE,
    STREAM
};

struct Config {
    RunMode mode;
    string task_path;
    string task_hash;
    string heuristic_config;
    size_t max_request_bytes;
    size_t max_requests_per_process;
    size_t max_registered_states;
    double max_score_seconds;

    Config()
        : mode(RunMode::NONE),
          heuristic_config(DEFAULT_HEURISTIC),
          max_request_bytes(1024 * 1024),
          max_requests_per_process(0),
          max_registered_states(0),
          max_score_seconds(30.0) {
    }
};

string basename_lower(const string &path) {
    size_t separator = path.find_last_of("/\\");
    string name = separator == string::npos
        ? path
        : path.substr(separator + 1);
    transform(name.begin(), name.end(), name.begin(),
              [](unsigned char ch) { return static_cast<char>(tolower(ch)); });
    return name;
}

void print_usage(ostream &out, const string &program_name) {
    out << "Usage: " << program_name
        << " --task PREPROCESSED_TASK (--once | --stream) [OPTIONS]\n"
        << "\n"
        << "JSONL trajectory scorer options:\n"
        << "  --task PATH                      C++ preprocessor output, not output.sas\n"
        << "  --task-hash HASH                 expected cache key; defaults to SAS SHA256\n"
        << "  --heuristic-config CONFIG        heuristic expression\n"
        << "  --max-request-bytes N            maximum JSONL line size (default 1048576)\n"
        << "  --max-score-seconds SECONDS      per-request soft deadline (default 30)\n"
        << "  --max-requests-per-process N     recommend recycle after N requests\n"
        << "  --max-registered-states N        recommend recycle at registry size N\n";
}

size_t parse_size(const string &option, const string &value, bool allow_zero) {
    if (value.empty() || value[0] == '-')
        throw runtime_error(option + " requires a non-negative integer");
    errno = 0;
    char *end = nullptr;
    unsigned long long parsed = strtoull(value.c_str(), &end, 10);
    if (errno == ERANGE || end != value.c_str() + value.size() ||
        parsed > numeric_limits<size_t>::max() || (!allow_zero && parsed == 0))
        throw runtime_error(option + " has an invalid integer value");
    return static_cast<size_t>(parsed);
}

double parse_positive_double(const string &option, const string &value) {
    errno = 0;
    char *end = nullptr;
    double parsed = strtod(value.c_str(), &end);
    if (errno == ERANGE || end != value.c_str() + value.size() ||
        !isfinite(parsed) || parsed <= 0)
        throw runtime_error(option + " has an invalid numeric value");
    return parsed;
}

string require_value(int argc, const char **argv, int &index) {
    if (index + 1 >= argc)
        throw runtime_error(string("missing value after ") + argv[index]);
    ++index;
    return argv[index];
}

Config parse_config(int argc, const char **argv) {
    Config config;
    for (int index = 1; index < argc; ++index) {
        string arg = argv[index];
        if (arg == "--trajectory-scorer") {
            continue;
        } else if (arg == "--once") {
            if (config.mode != RunMode::NONE)
                throw runtime_error("choose exactly one of --once and --stream");
            config.mode = RunMode::ONCE;
        } else if (arg == "--stream") {
            if (config.mode != RunMode::NONE)
                throw runtime_error("choose exactly one of --once and --stream");
            config.mode = RunMode::STREAM;
        } else if (arg == "--task") {
            config.task_path = require_value(argc, argv, index);
        } else if (arg == "--task-hash") {
            config.task_hash = require_value(argc, argv, index);
        } else if (arg == "--heuristic-config") {
            config.heuristic_config = require_value(argc, argv, index);
        } else if (arg == "--max-request-bytes") {
            config.max_request_bytes = parse_size(
                arg, require_value(argc, argv, index), false);
        } else if (arg == "--max-requests-per-process") {
            config.max_requests_per_process = parse_size(
                arg, require_value(argc, argv, index), true);
        } else if (arg == "--max-registered-states") {
            config.max_registered_states = parse_size(
                arg, require_value(argc, argv, index), true);
        } else if (arg == "--max-score-seconds") {
            config.max_score_seconds = parse_positive_double(
                arg, require_value(argc, argv, index));
        } else if (arg == "--help" || arg == "-h") {
            print_usage(cout, argc > 0 ? argv[0] : "nlm-score");
            throw runtime_error("");
        } else {
            throw runtime_error("unknown scorer option: " + arg);
        }
    }
    if (config.mode == RunMode::NONE)
        throw runtime_error("choose exactly one of --once and --stream");
    if (config.task_path.empty())
        throw runtime_error("--task is required");
    if (config.heuristic_config.empty())
        throw runtime_error("--heuristic-config must not be empty");
    return config;
}

class Sha256 {
    uint32_t state[8];
    uint8_t block[64];
    size_t block_size;
    uint64_t total_bits;

    static uint32_t rotate_right(uint32_t value, uint32_t bits) {
        return (value >> bits) | (value << (32 - bits));
    }

    void transform() {
        static const uint32_t K[64] = {
            0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
            0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
            0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
            0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
            0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
            0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
            0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
            0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
            0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
            0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
            0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
            0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
            0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
            0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
            0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
            0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
        };
        uint32_t words[64];
        for (size_t index = 0; index < 16; ++index) {
            size_t offset = index * 4;
            words[index] =
                (static_cast<uint32_t>(block[offset]) << 24) |
                (static_cast<uint32_t>(block[offset + 1]) << 16) |
                (static_cast<uint32_t>(block[offset + 2]) << 8) |
                static_cast<uint32_t>(block[offset + 3]);
        }
        for (size_t index = 16; index < 64; ++index) {
            uint32_t s0 = rotate_right(words[index - 15], 7) ^
                rotate_right(words[index - 15], 18) ^
                (words[index - 15] >> 3);
            uint32_t s1 = rotate_right(words[index - 2], 17) ^
                rotate_right(words[index - 2], 19) ^
                (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }

        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (size_t index = 0; index < 64; ++index) {
            uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                rotate_right(e, 25);
            uint32_t choice = (e & f) ^ ((~e) & g);
            uint32_t temp1 = h + s1 + choice + K[index] + words[index];
            uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                rotate_right(a, 22);
            uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            uint32_t temp2 = s0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }

public:
    Sha256()
        : state{0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19},
          block_size(0), total_bits(0) {
    }

    void update(const uint8_t *data, size_t size) {
        for (size_t index = 0; index < size; ++index) {
            block[block_size++] = data[index];
            if (block_size == 64) {
                transform();
                total_bits += 512;
                block_size = 0;
            }
        }
    }

    string finish() {
        total_bits += static_cast<uint64_t>(block_size) * 8;
        block[block_size++] = 0x80;
        if (block_size > 56) {
            while (block_size < 64)
                block[block_size++] = 0;
            transform();
            block_size = 0;
        }
        while (block_size < 56)
            block[block_size++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8)
            block[block_size++] =
                static_cast<uint8_t>((total_bits >> shift) & 0xff);
        transform();

        ostringstream out;
        out << hex << setfill('0');
        for (uint32_t word : state)
            out << setw(8) << word;
        return out.str();
    }
};

string compute_task_hash(const string &path) {
    ifstream input(path.c_str(), ios::binary);
    if (!input)
        throw runtime_error("cannot open task file for hashing: " + path);
    Sha256 hash;
    char buffer[64 * 1024];
    while (input) {
        input.read(buffer, sizeof(buffer));
        streamsize count = input.gcount();
        if (count > 0) {
            hash.update(
                reinterpret_cast<const uint8_t *>(buffer),
                static_cast<size_t>(count));
        }
    }
    if (!input.eof())
        throw runtime_error("failed while hashing task file: " + path);
    return "sas-sha256:" + hash.finish();
}

enum class LineReadStatus {
    LINE,
    END_OF_INPUT,
    TOO_LARGE
};

LineReadStatus read_bounded_line(
    istream &input, size_t max_bytes, string &line) {
    line.clear();
    bool too_large = false;
    char ch;
    while (input.get(ch)) {
        if (ch == '\n') {
            if (!too_large && !line.empty() && line.back() == '\r')
                line.pop_back();
            return too_large ? LineReadStatus::TOO_LARGE : LineReadStatus::LINE;
        }
        if (!too_large) {
            if (line.size() >= max_bytes) {
                too_large = true;
                line.clear();
            } else {
                line += ch;
            }
        }
    }
    if (too_large)
        return LineReadStatus::TOO_LARGE;
    if (!line.empty()) {
        if (line.back() == '\r')
            line.pop_back();
        return LineReadStatus::LINE;
    }
    return LineReadStatus::END_OF_INPUT;
}

int file_descriptor(FILE *file) {
#if defined(_WIN32)
    return _fileno(file);
#else
    return fileno(file);
#endif
}

int duplicate_descriptor(int descriptor) {
#if defined(_WIN32)
    return _dup(descriptor);
#else
    return dup(descriptor);
#endif
}

int replace_descriptor(int source, int destination) {
#if defined(_WIN32)
    return _dup2(source, destination);
#else
    return dup2(source, destination);
#endif
}

void close_descriptor(int descriptor) {
#if defined(_WIN32)
    _close(descriptor);
#else
    close(descriptor);
#endif
}

FILE *descriptor_to_file(int descriptor) {
#if defined(_WIN32)
    return _fdopen(descriptor, "w");
#else
    return fdopen(descriptor, "w");
#endif
}

struct FileCloser {
    void operator()(FILE *file) const {
        if (file)
            fclose(file);
    }
};

FILE *open_protocol_output() {
    cout.flush();
    cerr.flush();
    fflush(stdout);
    fflush(stderr);

    int stdout_descriptor = file_descriptor(stdout);
    int stderr_descriptor = file_descriptor(stderr);
    int protocol_descriptor = duplicate_descriptor(stdout_descriptor);
    if (protocol_descriptor < 0)
        throw runtime_error("failed to duplicate scorer stdout");
    if (replace_descriptor(stderr_descriptor, stdout_descriptor) < 0) {
        close_descriptor(protocol_descriptor);
        throw runtime_error("failed to isolate scorer protocol stdout");
    }

    FILE *protocol_output = descriptor_to_file(protocol_descriptor);
    if (!protocol_output) {
        replace_descriptor(protocol_descriptor, stdout_descriptor);
        close_descriptor(protocol_descriptor);
        throw runtime_error("failed to open scorer protocol stream");
    }

    // iostream and low-level writes to stdout now both go to stderr. Only
    // emit() writes to the duplicated original stdout protocol channel.
    cout.rdbuf(cerr.rdbuf());
    return protocol_output;
}

void emit(FILE *protocol_output, const string &message) {
    if (fwrite(message.data(), 1, message.size(), protocol_output) !=
            message.size() ||
        fputc('\n', protocol_output) == EOF ||
        fflush(protocol_output) == EOF) {
        throw runtime_error("failed to write scorer protocol message");
    }
}
}

bool is_scorer_invocation(int argc, const char **argv) {
    if (argc > 0) {
        string executable = basename_lower(argv[0]);
        if (executable == "nlm-score" || executable == "nlm-score.exe")
            return true;
    }
    for (int index = 1; index < argc; ++index) {
        if (string(argv[index]) == "--trajectory-scorer")
            return true;
    }
    return false;
}

int run(int argc, const char **argv) {
    Config config;
    try {
        config = parse_config(argc, argv);
    } catch (const exception &error) {
        if (string(error.what()).empty())
            return 0;
        cerr << "nlm-score: " << error.what() << '\n';
        print_usage(cerr, argc > 0 ? argv[0] : "nlm-score");
        return 2;
    }

    unique_ptr<FILE, FileCloser> protocol_output;

    try {
        protocol_output.reset(open_protocol_output());
        utils::register_event_handlers();
        if (config.task_hash.empty())
            config.task_hash = compute_task_hash(config.task_path);

        ifstream task_input(config.task_path.c_str(), ios::binary);
        if (!task_input)
            throw runtime_error("cannot open task file: " + config.task_path);
        read_everything(task_input);

        OptionParser heuristic_parser(config.heuristic_config, false);
        unique_ptr<Heuristic> heuristic(
            heuristic_parser.start_parsing<Heuristic *>());
        if (!heuristic)
            throw runtime_error("heuristic parser returned null");

        // Force lazy heuristic initialization before advertising readiness.
        // A stream client can therefore treat the ready message as a real
        // health check instead of discovering initialization failures on the
        // first training request.
        EvaluationContext warmup_context(
            g_initial_state(), 0.0, false, nullptr);
        warmup_context.get_result(heuristic.get());

        ActionChainEvaluator evaluator;
        if (config.mode == RunMode::STREAM) {
            emit(protocol_output.get(), trajectory_score_protocol::serialize_ready(
                config.task_hash, config.heuristic_config,
                config.max_request_bytes));
        }

        size_t handled_score_requests = 0;
        while (true) {
            string line;
            LineReadStatus line_status = read_bounded_line(
                cin, config.max_request_bytes, line);
            if (line_status == LineReadStatus::END_OF_INPUT)
                break;

            if (line_status == LineReadStatus::TOO_LARGE) {
                ScoreRequest request;
                emit(protocol_output.get(),
                     trajectory_score_protocol::serialize_error_response(
                         request, config.task_hash, "invalid_request",
                         "request exceeds max_request_bytes"));
                if (config.mode == RunMode::ONCE)
                    break;
                continue;
            }

            ParsedMessage parsed =
                trajectory_score_protocol::parse_message(line);
            if (parsed.kind == MessageKind::INVALID) {
                emit(protocol_output.get(),
                     trajectory_score_protocol::serialize_error_response(
                         parsed.request, config.task_hash, "invalid_request",
                         parsed.error));
                if (config.mode == RunMode::ONCE)
                    break;
                continue;
            }
            if (parsed.kind == MessageKind::SHUTDOWN) {
                emit(protocol_output.get(),
                     trajectory_score_protocol::serialize_shutdown_ack(
                         parsed.request.request_id, config.task_hash));
                break;
            }

            ++handled_score_requests;
            const ScoreRequest &request = parsed.request;
            if (request.task_hash != config.task_hash) {
                emit(protocol_output.get(),
                     trajectory_score_protocol::serialize_error_response(
                         request, config.task_hash, "task_mismatch",
                         "request task_hash does not match loaded task"));
                if (config.mode == RunMode::ONCE)
                    break;
                continue;
            }
            ActionChainEvaluationResult result;
            try {
                result = evaluator.evaluate(
                    g_initial_state(), request.actions, *heuristic,
                    config.max_score_seconds);
            } catch (const exception &error) {
                emit(protocol_output.get(),
                     trajectory_score_protocol::serialize_error_response(
                         request, config.task_hash, "internal_error",
                         error.what()));
                if (config.mode == RunMode::ONCE)
                    break;
                continue;
            }

            bool recycle_recommended = false;
            string recycle_reason;
            if (result.status == "scorer_timeout") {
                recycle_recommended = true;
                recycle_reason = "scorer_timeout";
            } else if (config.max_requests_per_process > 0 &&
                       handled_score_requests >=
                           config.max_requests_per_process) {
                recycle_recommended = true;
                recycle_reason = "max_requests_per_process";
            } else if (config.max_registered_states > 0 &&
                       result.registered_state_count >=
                           config.max_registered_states) {
                recycle_recommended = true;
                recycle_reason = "max_registered_states";
            }

            emit(protocol_output.get(),
                 trajectory_score_protocol::serialize_score_response(
                     request, config.task_hash, result,
                     recycle_recommended, recycle_reason));

            if (config.mode == RunMode::ONCE || recycle_recommended)
                break;
        }
        return 0;
    } catch (const ArgError &error) {
        cerr << "nlm-score: heuristic argument error: " << error << '\n';
        return 2;
    } catch (const ParseError &error) {
        cerr << "nlm-score: heuristic parse error: " << error << '\n';
        return 2;
    } catch (const exception &error) {
        cerr << "nlm-score: fatal initialization error: "
             << error.what() << '\n';
        return 2;
    }
}

}
