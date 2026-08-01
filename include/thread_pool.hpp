#pragma once

/**
 * Threading model
 * ---------------
 * A fixed-size worker pool (default: hardware_concurrency) owns a queue of
 * std::function<void()> jobs. Producer threads (main pipeline) push render /
 * crop tasks; workers pull and execute. Futures returned from submit() allow
 * the caller to wait for individual results and propagate exceptions.
 *
 * Why not unbounded std::async?
 * Over-subscribing FFmpeg/OpenCV workers thrashes disk and CPU caches. A
 * bounded pool keeps concurrency near the core count while still overlapping
 * I/O-bound encode jobs.
 */

#include <condition_variable>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

namespace autoclip {

class ThreadPool {
public:
    explicit ThreadPool(std::size_t n_threads = 0) {
        if (n_threads == 0) {
            n_threads = std::thread::hardware_concurrency();
            if (n_threads == 0) n_threads = 4;
        }
        workers_.reserve(n_threads);
        for (std::size_t i = 0; i < n_threads; ++i) {
            workers_.emplace_back([this] { worker_loop(); });
        }
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    ~ThreadPool() {
        {
            std::lock_guard lock(mutex_);
            stopping_ = true;
        }
        cv_.notify_all();
        for (auto& t : workers_) {
            if (t.joinable()) t.join();
        }
    }

    template <class F, class... Args>
    auto submit(F&& f, Args&&... args)
        -> std::future<std::invoke_result_t<F, Args...>> {
        using R = std::invoke_result_t<F, Args...>;

        auto task = std::make_shared<std::packaged_task<R()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...));

        std::future<R> fut = task->get_future();
        {
            std::lock_guard lock(mutex_);
            if (stopping_) {
                throw std::runtime_error("ThreadPool: submit on stopped pool");
            }
            jobs_.emplace([task]() { (*task)(); });
        }
        cv_.notify_one();
        return fut;
    }

    [[nodiscard]] std::size_t size() const noexcept { return workers_.size(); }

private:
    void worker_loop() {
        for (;;) {
            std::function<void()> job;
            {
                std::unique_lock lock(mutex_);
                cv_.wait(lock, [this] { return stopping_ || !jobs_.empty(); });
                if (stopping_ && jobs_.empty()) return;
                job = std::move(jobs_.front());
                jobs_.pop();
            }
            job();
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> jobs_;
    std::mutex mutex_;
    std::condition_variable cv_;
    bool stopping_ = false;
};

}  // namespace autoclip
