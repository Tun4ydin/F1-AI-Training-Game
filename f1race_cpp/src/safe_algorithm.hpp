#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace fai {

struct Parameter { double defaultValue{}, low{}, high{}; };
struct Outputs {
    double steering{}, throttle{}, brake{}, overtake{}, recharge{}, pitRequest{};
    int pitTyre{1};
};

class Program {
public:
    struct Expr;
    struct Statement;
    Program();
    explicit Program(const std::string& source);
    bool compile(const std::string& source);
    Outputs run(const std::unordered_map<std::string, double>& inputs,
                const std::map<std::string, double>& parameterValues) const;
    const std::map<std::string, Parameter>& parameters() const { return parameters_; }
    const std::string& error() const { return error_; }
    bool valid() const { return valid_; }
private:
    std::shared_ptr<std::vector<Statement>> statements_;
    std::map<std::string, Parameter> parameters_;
    std::string error_;
    bool valid_{};
};

} // namespace fai
