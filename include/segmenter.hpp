#pragma once

#include "types.hpp"

#include <vector>

namespace autoclip {

/**
 * Group word-level transcription into logical clip segments.
 *
 * Strategy:
 *   - Grow a segment toward ~target_sec (default 30s).
 *   - Prefer cuts where inter-word silence > pause_ms (default 400ms).
 *   - Force-cut at max_sec; discard segments shorter than min_sec.
 */
[[nodiscard]] std::vector<Segment> slice_segments(
    const std::vector<Word>& words,
    int target_sec = 30,
    int min_sec = 10,
    int max_sec = 60,
    int pause_ms = 400);

}  // namespace autoclip
