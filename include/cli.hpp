#pragma once

#include "types.hpp"

#include <optional>
#include <string>

namespace autoclip {

[[nodiscard]] std::optional<CliOptions> parse_cli(int argc, char** argv);
void print_usage(const char* argv0);

}  // namespace autoclip
