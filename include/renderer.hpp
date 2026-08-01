#pragma once

#include "types.hpp"

#include <filesystem>
#include <string>

namespace autoclip {

namespace fs = std::filesystem;

/** Extract 16 kHz mono PCM WAV via FFmpeg (-hwaccel auto). */
void extract_audio(const fs::path& input_video, const fs::path& output_wav);

struct RenderRequest {
    fs::path input_video;
    fs::path output_file;
    fs::path ass_file;
    Segment segment;
    CropRect crop;
    VideoEncoder encoder = VideoEncoder::X264;
};

/**
 * Render one vertical clip: crop + ASS burn-in + selected encoder.
 * Uses argv-style FFmpeg invocation (no shell) with -hwaccel auto.
 */
void render_clip(const RenderRequest& req);

}  // namespace autoclip
