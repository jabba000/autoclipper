#pragma once

/**
 * OpusClip-style semantic clip selection
 * --------------------------------------
 * 1. Split the word timeline into speech regions at every natural pause
 *    (never merge across long silence — that was the old bug).
 * 2. Subdivide long regions at pauses → sentence ends → duration.
 * 3. Score / filter candidates; select ~target_clips with timeline diversity.
 *
 * Verbose mode logs every Accepted / Rejected decision with a reason.
 */

#include "types.hpp"

#include <vector>

namespace autoclip {

struct SliceOptions {
    int target_sec = 30;
    int min_sec = 10;
    int max_sec = 60;
    int pause_ms = 400;
    int target_clips = 0;  // 0 = auto
    bool verbose = false;
};

[[nodiscard]] std::vector<Segment> slice_segments(const std::vector<Word>& words,
                                                  const SliceOptions& opt);

/** Back-compat wrapper used by older call sites. */
[[nodiscard]] inline std::vector<Segment> slice_segments(
    const std::vector<Word>& words, int target_sec = 30, int min_sec = 10,
    int max_sec = 60, int pause_ms = 400) {
    SliceOptions opt;
    opt.target_sec = target_sec;
    opt.min_sec = min_sec;
    opt.max_sec = max_sec;
    opt.pause_ms = pause_ms;
    return slice_segments(words, opt);
}

}  // namespace autoclip
