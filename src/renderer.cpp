#include "renderer.hpp"

#include "process.hpp"

#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>

namespace autoclip {

void extract_audio(const fs::path& input_video, const fs::path& output_wav) {
    std::vector<std::string> argv = {
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-hwaccel",
        "auto",
        "-i",
        input_video.string(),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_s16le",
        output_wav.string(),
    };
    run_process_checked(argv);
}

namespace {

std::string format_seconds(int64_t ms) {
    std::ostringstream oss;
    oss.setf(std::ios::fixed);
    oss.precision(3);
    oss << (static_cast<double>(ms) / 1000.0);
    return oss.str();
}

void append_encoder_args(std::vector<std::string>& argv, VideoEncoder enc) {
    switch (enc) {
        case VideoEncoder::Nvenc:
            argv.insert(argv.end(),
                        {"-c:v", "h264_nvenc", "-preset", "p4", "-cq", "23"});
            break;
        case VideoEncoder::Vaapi:
            // Caller must have VAAPI device; keep flags minimal & portable.
            argv.insert(argv.end(),
                        {"-c:v", "h264_vaapi", "-qp", "23"});
            break;
        case VideoEncoder::Qsv:
            argv.insert(argv.end(),
                        {"-c:v", "h264_qsv", "-global_quality", "23"});
            break;
        case VideoEncoder::X264:
        default:
            argv.insert(argv.end(), {"-c:v", "libx264", "-preset", "veryfast",
                                     "-crf", "23"});
            break;
    }
}

}  // namespace

void render_clip(const RenderRequest& req) {
    const std::string filt_ass = ffmpeg_filter_path(req.ass_file);

    // crop=W:H:X:Y then burn ASS. scale to 1080x1920 for consistent shorts.
    std::ostringstream vf;
    vf << "crop=" << req.crop.w << ':' << req.crop.h << ':' << req.crop.x
       << ':' << req.crop.y << ",scale=1080:1920:flags=lanczos,ass=" << filt_ass;

    std::vector<std::string> argv = {
        "ffmpeg",
        "-hide_banner",
        "-y",
        "-hwaccel",
        "auto",
        // Accurate seek: input-side -ss after -i is slower but frame-accurate;
        // for clip cutting we put -ss/-to before -i for speed, then re-encode.
        "-ss",
        format_seconds(req.segment.start_ms),
        "-to",
        format_seconds(req.segment.end_ms),
        "-i",
        req.input_video.string(),
        "-vf",
        vf.str(),
    };

    append_encoder_args(argv, req.encoder);

    argv.insert(argv.end(),
                {"-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
                 "-shortest", req.output_file.string()});

    run_process_checked(argv, [](std::string_view line) {
        // Surface FFmpeg progress lines that contain "time=".
        if (line.find("time=") != std::string_view::npos) {
            // Keep quiet by default; enable via piping verbose later.
            (void)line;
        }
    });
}

}  // namespace autoclip
