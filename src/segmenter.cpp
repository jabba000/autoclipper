#include "segmenter.hpp"

#include <algorithm>

namespace autoclip {

std::vector<Segment> slice_segments(const std::vector<Word>& words,
                                    int target_sec, int min_sec, int max_sec,
                                    int pause_ms) {
    std::vector<Segment> out;
    if (words.empty()) return out;

    const int64_t target_ms = static_cast<int64_t>(target_sec) * 1000;
    const int64_t min_ms = static_cast<int64_t>(min_sec) * 1000;
    const int64_t max_ms = static_cast<int64_t>(max_sec) * 1000;

    Segment cur;
    cur.start_ms = words.front().t0_ms;

    auto flush = [&](int64_t end_ms) {
        cur.end_ms = end_ms;
        const int64_t dur = cur.duration_ms();
        // Ignore segments outside [min, max] — max is enforced by force-cut,
        // but a trailing stub may still be short.
        if (dur >= min_ms && dur <= max_ms && !cur.words.empty()) {
            cur.index = static_cast<int>(out.size());
            out.push_back(cur);
        }
        cur = Segment{};
    };

    for (std::size_t i = 0; i < words.size(); ++i) {
        cur.words.push_back(words[i]);

        const bool is_last = (i + 1 == words.size());
        const int64_t gap =
            is_last ? 0 : (words[i + 1].t0_ms - words[i].t1_ms);
        const int64_t duration = words[i].t1_ms - cur.start_ms;

        const bool natural_pause = gap >= pause_ms;
        const bool near_target = duration >= target_ms;
        const bool over_max = duration >= max_ms;

        // Prefer cutting on silence once we've reached ~target length;
        // always force-cut at max to avoid mid-word overflow.
        if (is_last || over_max || (natural_pause && near_target) ||
            (natural_pause && duration >= min_ms &&
             duration >= target_ms - 5000)) {
            flush(words[i].t1_ms);
            if (!is_last) {
                cur.start_ms = words[i + 1].t0_ms;
            }
        }
    }

    // If the loop left a dangling open segment (shouldn't with is_last flush),
    // discard it — flush already ran for the last word.
    return out;
}

}  // namespace autoclip
