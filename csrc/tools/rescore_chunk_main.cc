#include <absl/flags/flag.h>
#include <absl/flags/parse.h>
#include <absl/log/globals.h>
#include <absl/log/initialize.h>
#include <absl/log/log.h>
#include <absl/strings/match.h>
#include <absl/strings/numbers.h>
#include <zlib.h>

#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "chess/board.h"
#include "proto/data_loader_config.pb.h"
#include "syzygy/syzygy.h"
#include "trainingdata/reader.h"
#include "trainingdata/rescorer.h"
#include "trainingdata/trainingdata_v6.h"
#include "trainingdata/writer.h"
#include "utils/exception.h"

ABSL_FLAG(std::string, chunk_path, "",
          "Path to the chunk file (.gz) that should be rescored.");
ABSL_FLAG(std::string, config_path, "",
          "Path to RootConfig textproto describing the data pipeline.");

namespace {

namespace fs = std::filesystem;
using ::lczero::training::ChunkRescorerConfig;

enum class TokenType {
  kIdentifier,
  kString,
  kNumber,
  kColon,
  kLBrace,
  kRBrace,
  kEnd,
};

struct Token {
  TokenType type = TokenType::kEnd;
  std::string text;
  bool is_integer = false;
};

class Tokenizer {
 public:
  explicit Tokenizer(std::string_view input) : input_(input) {}

  const Token& Peek() {
    if (!current_) current_ = NextToken();
    return *current_;
  }

  Token Consume() {
    Token token = Peek();
    current_.reset();
    return token;
  }

  Token Consume(TokenType expected, const char* context) {
    Token token = Peek();
    if (token.type != expected) {
      LOG(FATAL) << "Unexpected token while parsing " << context;
    }
    current_.reset();
    return token;
  }

  bool TryConsume(TokenType expected) {
    if (Peek().type != expected) return false;
    current_.reset();
    return true;
  }

 private:
  Token NextToken() {
    SkipWhitespaceAndComments();
    if (pos_ >= input_.size()) return Token{TokenType::kEnd, "", false};

    const char c = input_[pos_];
    if (c == '{') {
      ++pos_;
      return Token{TokenType::kLBrace, "{", false};
    }
    if (c == '}') {
      ++pos_;
      return Token{TokenType::kRBrace, "}", false};
    }
    if (c == ':') {
      ++pos_;
      return Token{TokenType::kColon, ":", false};
    }
    if (c == '"') return ParseString();

    if (IsNumberStart(c)) return ParseNumber();
    return ParseIdentifier();
  }

  void SkipWhitespaceAndComments() {
    while (pos_ < input_.size()) {
      char c = input_[pos_];
      if (std::isspace(static_cast<unsigned char>(c))) {
        ++pos_;
        continue;
      }
      if (c == '#') {
        SkipLine();
        continue;
      }
      if (c == '/' && pos_ + 1 < input_.size()) {
        char next = input_[pos_ + 1];
        if (next == '/') {
          pos_ += 2;
          SkipLine();
          continue;
        }
        if (next == '*') {
          pos_ += 2;
          SkipBlockComment();
          continue;
        }
      }
      break;
    }
  }

  void SkipLine() {
    while (pos_ < input_.size() && input_[pos_] != '\n') ++pos_;
    if (pos_ < input_.size()) ++pos_;
  }

  void SkipBlockComment() {
    while (pos_ + 1 < input_.size()) {
      if (input_[pos_] == '*' && input_[pos_ + 1] == '/') {
        pos_ += 2;
        return;
      }
      ++pos_;
    }
  }

  static bool IsNumberStart(char c) {
    return std::isdigit(static_cast<unsigned char>(c)) || c == '-' ||
           c == '+' || c == '.';
  }

  Token ParseIdentifier() {
    size_t start = pos_;
    while (pos_ < input_.size()) {
      char c = input_[pos_];
      if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '.' ||
          c == '-') {
        ++pos_;
      } else {
        break;
      }
    }
    return Token{TokenType::kIdentifier,
                 std::string(input_.substr(start, pos_ - start)), false};
  }

  Token ParseNumber() {
    size_t start = pos_;
    bool has_dot = false;
    bool has_exp = false;
    if (input_[pos_] == '+' || input_[pos_] == '-') ++pos_;
    while (pos_ < input_.size()) {
      char c = input_[pos_];
      if (std::isdigit(static_cast<unsigned char>(c))) {
        ++pos_;
        continue;
      }
      if (c == '.' && !has_dot) {
        has_dot = true;
        ++pos_;
        continue;
      }
      if ((c == 'e' || c == 'E') && !has_exp) {
        has_exp = true;
        ++pos_;
        if (pos_ < input_.size() &&
            (input_[pos_] == '+' || input_[pos_] == '-')) {
          ++pos_;
        }
        continue;
      }
      break;
    }
    bool is_integer = !has_dot && !has_exp;
    return Token{TokenType::kNumber,
                 std::string(input_.substr(start, pos_ - start)), is_integer};
  }

  Token ParseString() {
    ++pos_;
    std::string value;
    while (pos_ < input_.size()) {
      char c = input_[pos_++];
      if (c == '"') break;
      if (c == '\\') {
        if (pos_ >= input_.size()) break;
        char next = input_[pos_++];
        switch (next) {
          case 'n':
            value.push_back('\n');
            break;
          case 't':
            value.push_back('\t');
            break;
          case 'r':
            value.push_back('\r');
            break;
          case '\\':
            value.push_back('\\');
            break;
          case '"':
            value.push_back('"');
            break;
          default:
            value.push_back(next);
            break;
        }
        continue;
      }
      value.push_back(c);
    }
    return Token{TokenType::kString, std::move(value), false};
  }

  std::string_view input_;
  size_t pos_ = 0;
  std::optional<Token> current_;
};

struct Message;

struct Value {
  enum class Type { kString, kInt, kDouble, kBool, kMessage };
  Type type = Type::kString;
  std::string string_value;
  int64_t int_value = 0;
  double double_value = 0.0;
  bool bool_value = false;
  std::unique_ptr<Message> message_value;
};

struct Message {
  std::map<std::string, std::vector<Value>> fields;
};

class Parser {
 public:
  explicit Parser(std::string_view input) : tokenizer_(input) {}

  Message Parse() {
    Message message;
    ParseFields(&message, /*stop_at_rbrace=*/false);
    return message;
  }

 private:
  void ParseFields(Message* message, bool stop_at_rbrace) {
    while (true) {
      const Token& token = tokenizer_.Peek();
      if (stop_at_rbrace && token.type == TokenType::kRBrace) {
        tokenizer_.Consume();
        return;
      }
      if (token.type == TokenType::kEnd) return;
      if (token.type != TokenType::kIdentifier) {
        LOG(FATAL) << "Expected field name while parsing textproto.";
      }
      std::string field_name = tokenizer_.Consume().text;

      Value value;
      const Token& next = tokenizer_.Peek();
      if (next.type == TokenType::kColon) {
        tokenizer_.Consume();
        value = ParseValue();
      } else if (next.type == TokenType::kLBrace) {
        tokenizer_.Consume();
        value.type = Value::Type::kMessage;
        value.message_value = std::make_unique<Message>();
        ParseFields(value.message_value.get(), /*stop_at_rbrace=*/true);
      } else {
        LOG(FATAL) << "Expected ':' or '{' after field name '" << field_name
                   << "'.";
      }
      message->fields[field_name].push_back(std::move(value));
    }
  }

  Value ParseValue() {
    const Token& token = tokenizer_.Peek();
    Value value;
    switch (token.type) {
      case TokenType::kString: {
        value.type = Value::Type::kString;
        value.string_value = tokenizer_.Consume().text;
        break;
      }
      case TokenType::kNumber: {
        Token number = tokenizer_.Consume();
        if (number.is_integer) {
          value.type = Value::Type::kInt;
          if (!absl::SimpleAtoi(number.text, &value.int_value)) {
            LOG(FATAL) << "Failed to parse integer literal '" << number.text
                       << "'.";
          }
          value.double_value = static_cast<double>(value.int_value);
        } else {
          value.type = Value::Type::kDouble;
          if (!absl::SimpleAtod(number.text, &value.double_value)) {
            LOG(FATAL) << "Failed to parse float literal '" << number.text
                       << "'.";
          }
        }
        break;
      }
      case TokenType::kIdentifier: {
        std::string ident = tokenizer_.Consume().text;
        if (absl::EqualsIgnoreCase(ident, "true")) {
          value.type = Value::Type::kBool;
          value.bool_value = true;
        } else if (absl::EqualsIgnoreCase(ident, "false")) {
          value.type = Value::Type::kBool;
          value.bool_value = false;
        } else {
          value.type = Value::Type::kString;
          value.string_value = std::move(ident);
        }
        break;
      }
      case TokenType::kLBrace: {
        tokenizer_.Consume();
        value.type = Value::Type::kMessage;
        value.message_value = std::make_unique<Message>();
        ParseFields(value.message_value.get(), /*stop_at_rbrace=*/true);
        break;
      }
      default:
        LOG(FATAL) << "Unexpected token while reading value.";
    }
    return value;
  }

  Tokenizer tokenizer_;
};

const std::vector<Value>* FindField(const Message& message,
                                    const std::string& name) {
  auto it = message.fields.find(name);
  if (it == message.fields.end()) return nullptr;
  return &it->second;
}

const Value* FindSingleValue(const Message& message, const std::string& name) {
  const std::vector<Value>* values = FindField(message, name);
  if (values == nullptr || values->empty()) return nullptr;
  return &values->front();
}

const Message* FindSingleMessage(const Message& message,
                                 const std::string& name) {
  const std::vector<Value>* values = FindField(message, name);
  if (values == nullptr || values->empty()) return nullptr;
  const Value& value = values->front();
  if (value.type != Value::Type::kMessage) {
    LOG(FATAL) << "Field '" << name << "' is not a message.";
  }
  return value.message_value.get();
}

std::string GetString(const Value& value, std::string_view field_name) {
  if (value.type == Value::Type::kString) return value.string_value;
  if (value.type == Value::Type::kBool) {
    return value.bool_value ? "true" : "false";
  }
  if (value.type == Value::Type::kInt) {
    return std::to_string(value.int_value);
  }
  if (value.type == Value::Type::kDouble) {
    return std::to_string(value.double_value);
  }
  LOG(FATAL) << "Field '" << field_name << "' must be a scalar string.";
  return "";
}

int64_t GetInt(const Value& value, std::string_view field_name) {
  if (value.type == Value::Type::kInt) return value.int_value;
  if (value.type == Value::Type::kDouble) {
    return static_cast<int64_t>(value.double_value);
  }
  if (value.type == Value::Type::kString) {
    int64_t parsed = 0;
    if (absl::SimpleAtoi(value.string_value, &parsed)) return parsed;
  }
  LOG(FATAL) << "Field '" << field_name << "' must be an integer.";
  return 0;
}

double GetDouble(const Value& value, std::string_view field_name) {
  if (value.type == Value::Type::kDouble) return value.double_value;
  if (value.type == Value::Type::kInt) {
    return static_cast<double>(value.int_value);
  }
  if (value.type == Value::Type::kString) {
    double parsed = 0.0;
    if (absl::SimpleAtod(value.string_value, &parsed)) return parsed;
  }
  LOG(FATAL) << "Field '" << field_name << "' must be a float.";
  return 0.0;
}

ChunkRescorerConfig ExtractChunkRescorerConfig(const Message& root) {
  const Message* data_loader = FindSingleMessage(root, "data_loader");
  if (data_loader == nullptr) {
    LOG(FATAL) << "RootConfig is missing data_loader configuration.";
  }

  const std::vector<Value>* stages = FindField(*data_loader, "stage");
  if (stages == nullptr) {
    LOG(FATAL) << "Data loader configuration has no stage entries.";
  }

  const Message* chunk_rescorer_msg = nullptr;
  std::string stage_name;

  for (const Value& stage_value : *stages) {
    if (stage_value.type != Value::Type::kMessage) {
      LOG(FATAL) << "Stage entry is not a message.";
    }
    const Message& stage_message = *stage_value.message_value;
    const Message* candidate =
        FindSingleMessage(stage_message, "chunk_rescorer");
    if (candidate == nullptr) continue;
    if (chunk_rescorer_msg != nullptr) {
      LOG(FATAL) << "Multiple chunk_rescorer stages found in configuration.";
    }
    chunk_rescorer_msg = candidate;
    if (const Value* name_value = FindSingleValue(stage_message, "name")) {
      stage_name = GetString(*name_value, "name");
    }
  }

  if (chunk_rescorer_msg == nullptr) {
    LOG(FATAL) << "No chunk_rescorer stage found in data loader configuration.";
  }

  ChunkRescorerConfig config;

  if (const Value* v = FindSingleValue(*chunk_rescorer_msg, "syzygy_paths")) {
    config.set_syzygy_paths(GetString(*v, "syzygy_paths"));
  }
  if (const Value* v = FindSingleValue(*chunk_rescorer_msg, "dist_temp")) {
    config.set_dist_temp(static_cast<float>(GetDouble(*v, "dist_temp")));
  }
  if (const Value* v = FindSingleValue(*chunk_rescorer_msg, "dist_offset")) {
    config.set_dist_offset(static_cast<float>(GetDouble(*v, "dist_offset")));
  }
  if (const Value* v = FindSingleValue(*chunk_rescorer_msg, "dtz_boost")) {
    config.set_dtz_boost(static_cast<float>(GetDouble(*v, "dtz_boost")));
  }
  if (const Value* v =
          FindSingleValue(*chunk_rescorer_msg, "new_input_format")) {
    config.set_new_input_format(
        static_cast<int32_t>(GetInt(*v, "new_input_format")));
  }
  if (const Value* v =
          FindSingleValue(*chunk_rescorer_msg, "deblunder_threshold")) {
    config.set_deblunder_threshold(
        static_cast<float>(GetDouble(*v, "deblunder_threshold")));
  }
  if (const Value* v =
          FindSingleValue(*chunk_rescorer_msg, "deblunder_width")) {
    config.set_deblunder_width(
        static_cast<float>(GetDouble(*v, "deblunder_width")));
  }
  if (const Value* v = FindSingleValue(*chunk_rescorer_msg, "threads")) {
    config.set_threads(static_cast<uint64_t>(GetInt(*v, "threads")));
  }
  if (const Value* v = FindSingleValue(*chunk_rescorer_msg, "queue_capacity")) {
    config.set_queue_capacity(
        static_cast<uint64_t>(GetInt(*v, "queue_capacity")));
  }
  if (const Value* v = FindSingleValue(*chunk_rescorer_msg, "input")) {
    config.set_input(GetString(*v, "input"));
  }

  if (!stage_name.empty()) {
    LOG(INFO) << "Using chunk_rescorer stage '" << stage_name << "'.";
  }

  return config;
}

std::string ReadFile(const fs::path& path) {
  std::ifstream stream(path, std::ios::in | std::ios::binary);
  if (!stream.is_open()) {
    LOG(FATAL) << "Failed to open file: " << path.string();
  }
  std::string contents((std::istreambuf_iterator<char>(stream)),
                       std::istreambuf_iterator<char>());
  return contents;
}

std::vector<lczero::V6TrainingData> ReadChunkFrames(const fs::path& path) {
  std::vector<lczero::V6TrainingData> frames;
  lczero::TrainingDataReader reader(path.string());
  lczero::V6TrainingData frame;
  while (reader.ReadChunk(&frame)) {
    frames.push_back(frame);
  }
  return frames;
}

void WriteChunkFrames(const fs::path& path,
                      const std::vector<lczero::V6TrainingData>& frames) {
  lczero::TrainingDataWriter writer(path.string());
  for (const auto& frame : frames) {
    writer.WriteChunk(frame);
  }
  writer.Finalize();
}

fs::path BuildOutputPath(const fs::path& input_path) {
  fs::path directory = input_path.parent_path();
  fs::path stem = input_path.stem();
  fs::path filename = stem;
  filename += "_rescored.gz";
  return directory / filename;
}

}  // namespace

int main(int argc, char** argv) {
  absl::InitializeLog();
  absl::ParseCommandLine(argc, argv);
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const std::string chunk_path_flag = absl::GetFlag(FLAGS_chunk_path);
  if (chunk_path_flag.empty()) {
    LOG(FATAL) << "--chunk_path flag is required.";
  }
  const std::string config_path_flag = absl::GetFlag(FLAGS_config_path);
  if (config_path_flag.empty()) {
    LOG(FATAL) << "--config_path flag is required.";
  }

  const fs::path chunk_path(chunk_path_flag);
  const fs::path config_path(config_path_flag);

  const std::string config_text = ReadFile(config_path);
  Parser parser(config_text);
  const Message root_message = parser.Parse();
  const ChunkRescorerConfig config = ExtractChunkRescorerConfig(root_message);

  LOG(INFO) << "Reading chunk from " << chunk_path.string();
  std::vector<lczero::V6TrainingData> frames;
  try {
    frames = ReadChunkFrames(chunk_path);
  } catch (const lczero::Exception& exception) {
    LOG(FATAL) << "Failed to read chunk: " << exception.what();
  }
  LOG(INFO) << "Loaded " << frames.size() << " frame(s) from chunk.";
  if (frames.empty()) {
    LOG(WARNING) << "Chunk contains no frames; writing empty output.";
    try {
      WriteChunkFrames(BuildOutputPath(chunk_path), frames);
    } catch (const lczero::Exception& exception) {
      LOG(FATAL) << "Failed to write rescored chunk: " << exception.what();
    }
    return 0;
  }

  lczero::InitializeMagicBitboards();

  if (config.has_deblunder_threshold() && config.has_deblunder_width()) {
    lczero::RescorerDeblunderSetup(config.deblunder_threshold(),
                                   config.deblunder_width());
  }

  lczero::SyzygyTablebase tablebase;
  if (!config.syzygy_paths().empty()) {
    LOG(INFO) << "Initializing Syzygy tablebases from '"
              << config.syzygy_paths() << "'.";
    const std::string syzygy_paths(config.syzygy_paths());
    if (!tablebase.init(syzygy_paths)) {
      LOG(WARNING) << "Failed to initialize Syzygy tablebases.";
    }
  }

  LOG(INFO) << "Rescoring chunk with dist_temp=" << config.dist_temp()
            << ", dist_offset=" << config.dist_offset()
            << ", dtz_boost=" << config.dtz_boost()
            << ", new_input_format=" << config.new_input_format() << ".";

  try {
    frames = lczero::RescoreTrainingData(
        std::move(frames), &tablebase, config.dist_temp(), config.dist_offset(),
        config.dtz_boost(), config.new_input_format());
  } catch (const lczero::Exception& exception) {
    LOG(FATAL) << "Failed to rescore chunk: " << exception.what();
  }

  const fs::path output_path = BuildOutputPath(chunk_path);
  LOG(INFO) << "Writing rescored chunk to " << output_path.string();
  try {
    WriteChunkFrames(output_path, frames);
  } catch (const lczero::Exception& exception) {
    LOG(FATAL) << "Failed to write rescored chunk: " << exception.what();
  }
  LOG(INFO) << "Completed rescoring of chunk.";

  return 0;
}
