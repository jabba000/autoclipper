#include "process.hpp"

#include <array>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <spawn.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <sstream>

extern char** environ;

namespace autoclip {

std::string shell_quote(std::string_view s) {
    std::string out;
    out.reserve(s.size() + 2);
    out.push_back('\'');
    for (char c : s) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out.push_back(c);
        }
    }
    out.push_back('\'');
    return out;
}

std::string ffmpeg_filter_path(const fs::path& p) {
    // Absolute, forward-slash path; escape \ and : for filtergraph parsing.
    fs::path abs = fs::absolute(p).lexically_normal();
    std::string s = abs.string();
    std::replace(s.begin(), s.end(), '\\', '/');
    std::string out;
    out.reserve(s.size() + 8);
    for (char c : s) {
        if (c == '\\' || c == ':' || c == '\'' || c == '[' || c == ']') {
            out.push_back('\\');
        }
        out.push_back(c);
    }
    return out;
}

namespace {

void set_cloexec(int fd) {
    int flags = fcntl(fd, F_GETFD);
    if (flags >= 0) {
        fcntl(fd, F_SETFD, flags | FD_CLOEXEC);
    }
}

void drain_fd(int fd, std::string& sink,
              const std::function<void(std::string_view)>& on_line,
              std::string& line_buf) {
    std::array<char, 4096> buf{};
    for (;;) {
        ssize_t n = ::read(fd, buf.data(), buf.size());
        if (n > 0) {
            sink.append(buf.data(), static_cast<size_t>(n));
            if (on_line) {
                line_buf.append(buf.data(), static_cast<size_t>(n));
                std::size_t pos;
                while ((pos = line_buf.find('\n')) != std::string::npos) {
                    on_line(std::string_view(line_buf.data(), pos));
                    line_buf.erase(0, pos + 1);
                }
            }
            continue;
        }
        if (n == 0) break;
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) break;
        break;
    }
}

}  // namespace

ProcessResult run_process(
    const std::vector<std::string>& argv,
    const std::function<void(std::string_view)>& on_stderr_line) {
    if (argv.empty()) {
        throw std::invalid_argument("run_process: empty argv");
    }

    int out_pipe[2] = {-1, -1};
    int err_pipe[2] = {-1, -1};
    if (pipe(out_pipe) != 0 || pipe(err_pipe) != 0) {
        throw std::runtime_error(std::string("pipe: ") + std::strerror(errno));
    }

    posix_spawn_file_actions_t actions;
    posix_spawn_file_actions_init(&actions);
    posix_spawn_file_actions_addclose(&actions, out_pipe[0]);
    posix_spawn_file_actions_addclose(&actions, err_pipe[0]);
    posix_spawn_file_actions_adddup2(&actions, out_pipe[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, err_pipe[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, out_pipe[1]);
    posix_spawn_file_actions_addclose(&actions, err_pipe[1]);

    std::vector<char*> c_argv;
    c_argv.reserve(argv.size() + 1);
    for (const auto& a : argv) {
        c_argv.push_back(const_cast<char*>(a.c_str()));
    }
    c_argv.push_back(nullptr);

    pid_t pid = 0;
    const int spawn_rc = posix_spawnp(&pid, argv[0].c_str(), &actions, nullptr,
                                      c_argv.data(), environ);
    posix_spawn_file_actions_destroy(&actions);
    ::close(out_pipe[1]);
    ::close(err_pipe[1]);

    if (spawn_rc != 0) {
        ::close(out_pipe[0]);
        ::close(err_pipe[0]);
        throw std::runtime_error(std::string("posix_spawnp(") + argv[0] +
                                 "): " + std::strerror(spawn_rc));
    }

    // Non-blocking reads so we can poll both pipes without deadlock.
    fcntl(out_pipe[0], F_SETFL, O_NONBLOCK);
    fcntl(err_pipe[0], F_SETFL, O_NONBLOCK);
    set_cloexec(out_pipe[0]);
    set_cloexec(err_pipe[0]);

    ProcessResult result;
    std::string err_line_buf;
    bool out_open = true;
    bool err_open = true;

    while (out_open || err_open) {
        pollfd fds[2]{};
        nfds_t nfds = 0;
        int out_idx = -1;
        int err_idx = -1;
        if (out_open) {
            fds[nfds].fd = out_pipe[0];
            fds[nfds].events = POLLIN | POLLHUP;
            out_idx = static_cast<int>(nfds++);
        }
        if (err_open) {
            fds[nfds].fd = err_pipe[0];
            fds[nfds].events = POLLIN | POLLHUP;
            err_idx = static_cast<int>(nfds++);
        }

        int pr = ::poll(fds, nfds, 200);
        if (pr < 0) {
            if (errno == EINTR) continue;
            break;
        }

        if (out_idx >= 0 && (fds[out_idx].revents & (POLLIN | POLLHUP | POLLERR))) {
            const size_t before = result.stdout_data.size();
            drain_fd(out_pipe[0], result.stdout_data, {}, err_line_buf);
            if (result.stdout_data.size() == before &&
                (fds[out_idx].revents & (POLLHUP | POLLERR))) {
                out_open = false;
            }
        }
        if (err_idx >= 0 && (fds[err_idx].revents & (POLLIN | POLLHUP | POLLERR))) {
            const size_t before = result.stderr_data.size();
            drain_fd(err_pipe[0], result.stderr_data, on_stderr_line, err_line_buf);
            if (result.stderr_data.size() == before &&
                (fds[err_idx].revents & (POLLHUP | POLLERR))) {
                err_open = false;
            }
        }
    }

    if (!err_line_buf.empty() && on_stderr_line) {
        on_stderr_line(err_line_buf);
    }

    ::close(out_pipe[0]);
    ::close(err_pipe[0]);

    int status = 0;
    while (::waitpid(pid, &status, 0) < 0) {
        if (errno != EINTR) {
            throw std::runtime_error(std::string("waitpid: ") + std::strerror(errno));
        }
    }
    if (WIFEXITED(status)) {
        result.exit_code = WEXITSTATUS(status);
    } else if (WIFSIGNALED(status)) {
        result.exit_code = 128 + WTERMSIG(status);
    }
    return result;
}

void run_process_checked(
    const std::vector<std::string>& argv,
    const std::function<void(std::string_view)>& on_stderr_line) {
    auto res = run_process(argv, on_stderr_line);
    if (!res.ok()) {
        std::ostringstream oss;
        oss << "Command failed (exit " << res.exit_code << "):";
        for (const auto& a : argv) oss << ' ' << a;
        // Prefer stderr; fall back to stdout. Truncate for readability.
        std::string detail = res.stderr_data.empty() ? res.stdout_data : res.stderr_data;
        if (detail.size() > 2000) {
            detail = detail.substr(detail.size() - 2000);
        }
        if (!detail.empty()) {
            oss << "\n---\n" << detail;
        }
        throw std::runtime_error(oss.str());
    }
}

}  // namespace autoclip
