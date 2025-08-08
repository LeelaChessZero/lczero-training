#pragma once

#include <absl/strings/str_cat.h>

#include <string>
#include <string_view>

namespace lczero {

class MetricPrinter {
 public:
  virtual ~MetricPrinter() = default;
  virtual void StartGroup(std::string_view group_name) = 0;
  virtual void Print(std::string_view metric_name,
                     const std::string& value) = 0;
  virtual void Print(std::string_view metric_name, size_t value) {
    Print(metric_name, std::to_string(value));
  }
  virtual void EndGroup() = 0;
};

class StringMetricPrinter : public MetricPrinter {
 public:
  StringMetricPrinter(std::string* output) : output_(output) {}
  void StartGroup(std::string_view group_name) override {
    if (!first_group_) absl::StrAppend(output_, ", ");
    absl::StrAppend(output_, group_name, "={");
    first_group_ = false;
    first_metric_ = true;
  }

  void Print(std::string_view metric_name, const std::string& value) override {
    if (!first_metric_) absl::StrAppend(output_, ", ");
    absl::StrAppend(output_, metric_name, "=", value);
    first_metric_ = false;
    first_group_ = false;
  }

  void EndGroup() override { absl::StrAppend(output_, "}"); }

 private:
  std::string* output_;
  bool first_metric_ = true;
  bool first_group_ = true;
};

template <typename T>
std::string MetricToString(const T& metric) {
  std::string result;
  StringMetricPrinter printer(&result);
  metric.Print(printer);
  return result;
}

}  // namespace lczero