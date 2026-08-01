#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace autoclip {

namespace fs = std::filesystem;

struct ProcessResult {
    int exit_code = -1;
    std::string stdout_data;
    std::string stderr_data;

    [[nodiscard]] bool ok() const noexcept { return exit_code == 0; }
};

/**
 * Escape a filesystem path for safe embedding inside a POSIX shell
 * single-quoted argument. Uses '\'' to break/resume the quote.
 */
[[nodiscard]] std::string shell_quote(std::string_view s);

/**
 * Escape a path for use inside an FFmpeg filtergraph option value
 * (e.g. ass=...). Colons and escaping backslashes must be escaped.
 */
[[nodiscard]] std::string ffmpeg_filter_path(const fs::path& p);

/**
 * Run an external command via posix_spawn + pipes.
 * Captures stdout/stderr separately. Optionally invokes on_stderr_line
 * for each complete stderr line (used for FFmpeg progress).
 *
 * Does NOT use std::system. argv[0] is the executable path/name.
 */
ProcessResult run_process(
    const std::vector<std::string>& argv,
    const std::function<void(std::string_view)>& on_stderr_line = {});

/**
 * Convenience: run argv and throw std::runtime_error on non-zero exit.
 * Includes truncated stderr in the exception message.
 */
void run_process_checked(
    const std::vector<std::string>& argv,
    const std::function<void(std::string_view)>& on_stderr_line = {});

}  // namespace autoclip
