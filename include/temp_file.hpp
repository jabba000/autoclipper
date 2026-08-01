#pragma once

#include <filesystem>
#include <string>
#include <utility>
#include <vector>

namespace autoclip {

namespace fs = std::filesystem;

/**
 * RAII guard that deletes a file (or directory tree) on destruction.
 * Exception-safe: cleanup runs even if the owning scope unwinds.
 */
class TempPath {
public:
    TempPath() = default;

    explicit TempPath(fs::path path, bool is_directory = false)
        : path_(std::move(path)), is_directory_(is_directory) {}

    TempPath(const TempPath&) = delete;
    TempPath& operator=(const TempPath&) = delete;

    TempPath(TempPath&& other) noexcept
        : path_(std::move(other.path_)),
          is_directory_(other.is_directory_),
          active_(other.active_) {
        other.active_ = false;
    }

    TempPath& operator=(TempPath&& other) noexcept {
        if (this != &other) {
            cleanup();
            path_ = std::move(other.path_);
            is_directory_ = other.is_directory_;
            active_ = other.active_;
            other.active_ = false;
        }
        return *this;
    }

    ~TempPath() { cleanup(); }

    [[nodiscard]] const fs::path& path() const noexcept { return path_; }
    [[nodiscard]] std::string string() const { return path_.string(); }
    [[nodiscard]] explicit operator bool() const noexcept { return active_; }

    /** Release ownership without deleting (e.g. keep a final artifact). */
    fs::path release() noexcept {
        active_ = false;
        return std::move(path_);
    }

    void reset(fs::path path = {}, bool is_directory = false) {
        cleanup();
        path_ = std::move(path);
        is_directory_ = is_directory;
        active_ = !path_.empty();
    }

private:
    void cleanup() noexcept {
        if (!active_ || path_.empty()) return;
        std::error_code ec;
        if (is_directory_) {
            fs::remove_all(path_, ec);
        } else {
            fs::remove(path_, ec);
        }
        active_ = false;
    }

    fs::path path_;
    bool is_directory_ = false;
    bool active_ = true;
};

/**
 * Aggregates multiple TempPath objects so a whole working set is cleaned up
 * when the owner goes out of scope.
 */
class TempSet {
public:
    TempPath& add_file(fs::path p) {
        items_.emplace_back(std::move(p), false);
        return items_.back();
    }

    TempPath& add_dir(fs::path p) {
        items_.emplace_back(std::move(p), true);
        return items_.back();
    }

    void clear() { items_.clear(); }

private:
    std::vector<TempPath> items_;
};

}  // namespace autoclip
