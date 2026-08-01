#pragma once

/**
 * TikTok / OpusClip-style ASS subtitle generation
 * -----------------------------------------------
 * Emits Advanced SubStation Alpha with:
 *   - Arial Black, yellow karaoke highlight, thick black outline
 *   - PlayRes 1080x1920 matching the 9:16 crop
 *   - Placement ~72% down the frame (Alignment 2 + MarginV) — below faces
 *   - Chunks of preferably 2 words (max 3); one Dialogue event per chunk so
 *     the subtitle only changes after the whole chunk has been spoken
 *   - Karaoke \\k within the event (no per-word Dialogue flicker)
 *
 * Timestamps are relative to the segment start (clip-local timeline).
 */

#include "types.hpp"

#include <filesystem>

namespace autoclip {

namespace fs = std::filesystem;

/** Write karaoke ASS for one segment. Returns the path written. */
fs::path write_karaoke_ass(const Segment& seg, const fs::path& out_path);

}  // namespace autoclip
