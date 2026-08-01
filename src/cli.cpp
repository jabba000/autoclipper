#include "cli.hpp"

#include <getopt.h>

#include <cstring>
#include <iostream>
#include <stdexcept>

namespace autoclip {

void print_usage(const char* argv0) {
    std::cerr
        << "AutoClipper — local OpusClip-style vertical clip generator\n\n"
        << "Usage: " << argv0 << " -i <video> [options]\n\n"
        << "Required:\n"
        << "  -i, --input PATH          Input video file\n\n"
        << "Options:\n"
        << "  -o, --output DIR          Output directory (default: ./output)\n"
        << "  -m, --model PATH          Whisper ggml model (default: extra/ggml-base.bin)\n"
        << "  -e, --encoder NAME        Video encoder: x264|nvenc|vaapi|qsv (default: x264)\n"
        << "      --face-proto PATH     OpenCV DNN Caffe prototxt\n"
        << "      --face-model PATH     OpenCV DNN Caffe weights (.caffemodel)\n"
        << "  -l, --language CODE       Whisper language (default: uk)\n"
        << "  -t, --threads N           Worker threads (default: hardware_concurrency)\n"
        << "      --target-sec N        Target segment length (default: 30)\n"
        << "      --min-sec N           Min segment length (default: 10)\n"
        << "      --max-sec N           Max segment length (default: 60)\n"
        << "      --pause-sec F         Silence cut threshold seconds (default: 0.4)\n"
        << "  -v, --verbose             Verbose logging\n"
        << "  -h, --help                Show this help\n\n"
        << "OpenCV DNN models (res10_300x300 SSD):\n"
        << "  Place deploy.prototxt + res10_300x300_ssd_iter_140000_fp16.caffemodel\n"
        << "  under extra/, or pass --face-proto / --face-model. Without them the\n"
        << "  cropper falls back to a centered 9:16 window.\n";
}

static VideoEncoder parse_encoder(const std::string& s) {
    if (s == "x264" || s == "libx264") return VideoEncoder::X264;
    if (s == "nvenc" || s == "h264_nvenc") return VideoEncoder::Nvenc;
    if (s == "vaapi" || s == "h264_vaapi") return VideoEncoder::Vaapi;
    if (s == "qsv" || s == "h264_qsv") return VideoEncoder::Qsv;
    throw std::runtime_error("Unknown encoder: " + s +
                             " (expected x264|nvenc|vaapi|qsv)");
}

std::optional<CliOptions> parse_cli(int argc, char** argv) {
    CliOptions opt;

    static const option long_opts[] = {
        {"input", required_argument, nullptr, 'i'},
        {"output", required_argument, nullptr, 'o'},
        {"model", required_argument, nullptr, 'm'},
        {"encoder", required_argument, nullptr, 'e'},
        {"face-proto", required_argument, nullptr, 1000},
        {"face-model", required_argument, nullptr, 1001},
        {"language", required_argument, nullptr, 'l'},
        {"threads", required_argument, nullptr, 't'},
        {"target-sec", required_argument, nullptr, 1002},
        {"min-sec", required_argument, nullptr, 1003},
        {"max-sec", required_argument, nullptr, 1004},
        {"pause-sec", required_argument, nullptr, 1005},
        {"verbose", no_argument, nullptr, 'v'},
        {"help", no_argument, nullptr, 'h'},
        {nullptr, 0, nullptr, 0},
    };

    int c;
    optind = 1;
    while ((c = getopt_long(argc, argv, "i:o:m:e:l:t:vh", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'i': opt.input_video = optarg; break;
            case 'o': opt.output_dir = optarg; break;
            case 'm': opt.whisper_model = optarg; break;
            case 'e': opt.encoder = parse_encoder(optarg); break;
            case 'l': opt.language = optarg; break;
            case 't': opt.threads = std::stoi(optarg); break;
            case 'v': opt.verbose = true; break;
            case 'h': print_usage(argv[0]); return std::nullopt;
            case 1000: opt.face_proto = optarg; break;
            case 1001: opt.face_model = optarg; break;
            case 1002: opt.target_seg_sec = std::stoi(optarg); break;
            case 1003: opt.min_seg_sec = std::stoi(optarg); break;
            case 1004: opt.max_seg_sec = std::stoi(optarg); break;
            case 1005: opt.pause_sec = std::stod(optarg); break;
            default:
                print_usage(argv[0]);
                return std::nullopt;
        }
    }

    // Positional fallback: autoclipper <input> [model] [out_dir]
    if (opt.input_video.empty() && optind < argc) {
        opt.input_video = argv[optind++];
    }
    if (optind < argc) opt.whisper_model = argv[optind++];
    if (optind < argc) opt.output_dir = argv[optind++];
    if (optind < argc) opt.face_proto = argv[optind++];
    if (optind < argc) opt.face_model = argv[optind++];

    if (opt.input_video.empty()) {
        print_usage(argv[0]);
        return std::nullopt;
    }
    return opt;
}

}  // namespace autoclip
