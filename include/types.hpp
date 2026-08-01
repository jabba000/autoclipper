#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace autoclip {

struct Word {
    std::string text;
    int64_t t0_ms = 0;  // absolute timeline, milliseconds
    int64_t t1_ms = 0;
};

struct Segment {
    int index = 0;
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    std::vector<Word> words;

    [[nodiscard]] int64_t duration_ms() const noexcept {
        return end_ms - start_ms;
    }
};

enum class VideoEncoder {
    X264,
    Nvenc,
    Vaapi,
    Qsv,
};

[[nodiscard]] inline const char* encoder_name(VideoEncoder e) noexcept {
    switch (e) {
        case VideoEncoder::Nvenc: return "h264_nvenc";
        case VideoEncoder::Vaapi: return "h264_vaapi";
        case VideoEncoder::Qsv:   return "h264_qsv";
        case VideoEncoder::X264:
        default:                  return "libx264";
    }
}

struct CropRect {
    int x = 0;
    int y = 0;
    int w = 0;
    int h = 0;
};

struct CliOptions {
    std::string input_video;
    std::string output_dir = "./output";
    std::string whisper_model = "extra/ggml-base.bin";
    std::string face_proto = "extra/deploy.prototxt";
    std::string face_model = "extra/res10_300x300_ssd_iter_140000_fp16.caffemodel";
    VideoEncoder encoder = VideoEncoder::X264;
    std::string language = "uk";
    int threads = 0;           // 0 = auto
    int target_seg_sec = 30;
    int min_seg_sec = 10;
    int max_seg_sec = 60;
    double pause_sec = 0.4;
    bool verbose = false;
};

}  // namespace autoclip
