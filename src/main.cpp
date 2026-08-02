/**
 * AutoClipper — local OpusClip-style vertical short-form clipper
 * ==============================================================
 *
 * Pipeline
 * --------
 *  1. Extract 16 kHz mono WAV via FFmpeg (posix_spawn, -hwaccel auto).
 *  2. Transcribe with whisper.cpp (Ukrainian, word-level timestamps).
 *  3. Slice into ~30 s segments at natural pauses (>0.4 s silence).
 *  4. Per segment (ThreadPool workers):
 *       a. OpenCV DNN face track → smoothed 9:16 crop
 *       b. Generate TikTok-style karaoke ASS
 *       c. FFmpeg crop + ass burn-in + selected encoder
 *  5. TempPath RAII deletes WAV / ASS / scratch files on scope exit.
 *
 * Threading model
 * ---------------
 * Transcription is single-threaded on the Whisper context (not re-entrant).
 * After slicing, a bounded ThreadPool (default: hardware_concurrency) runs
 * face-track + render jobs. Each worker owns (or shares via mutex) a
 * FaceTracker; FFmpeg encodes run out-of-process so CPU/GPU overlap freely.
 *
 * OpenCV DNN model files
 * ----------------------
 *   --face-proto  extra/deploy.prototxt
 *   --face-model  extra/res10_300x300_ssd_iter_140000_fp16.caffemodel
 * Download from OpenCV's face_detector sample if missing. Without them the
 * cropper falls back to a centered 9:16 window.
 */

#include "ass_generator.hpp"
#include "cli.hpp"
#include "face_tracker.hpp"
#include "renderer.hpp"
#include "segmenter.hpp"
#include "temp_file.hpp"
#include "thread_pool.hpp"
#include "types.hpp"
#include "whisper_transcriber.hpp"

#include <unistd.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    using namespace autoclip;

    auto parsed = parse_cli(argc, argv);
    if (!parsed) return (argc > 1 && std::string(argv[1]) == "-h") ? 0 : 1;
    const CliOptions opt = std::move(*parsed);

    if (!fs::exists(opt.input_video)) {
        std::cerr << "[Fatal] Input not found: " << opt.input_video << '\n';
        return 1;
    }
    if (!fs::exists(opt.whisper_model)) {
        std::cerr << "[Fatal] Whisper model not found: " << opt.whisper_model << '\n';
        return 1;
    }

    fs::create_directories(opt.output_dir);
    const fs::path out_dir = fs::absolute(opt.output_dir);
    const fs::path input = fs::absolute(opt.input_video);

    // Scratch directory — wiped by TempPath even on exceptions.
    TempPath scratch_dir(out_dir / (".autoclip_tmp_" + std::to_string(::getpid())),
                         /*is_directory=*/true);
    fs::create_directories(scratch_dir.path());

    try {
        const auto t0 = std::chrono::steady_clock::now();

        // --- Step 1: audio + transcription ---------------------------------
        TempPath wav(scratch_dir.path() / "audio_16k.wav");
        std::cout << "[1/5] Extracting 16 kHz audio...\n";
        extract_audio(input, wav.path());

        std::cout << "[2/5] Loading Whisper (" << opt.whisper_model << ")...\n";
        WhisperContext wctx(opt.whisper_model);
        std::cout << "[2/5] Transcribing (" << opt.language << ")...\n";
        const auto words =
            transcribe(wctx, wav.path(), opt.language, opt.threads);
        std::cout << "       " << words.size() << " words with timestamps\n";

        // WAV no longer needed — release early to free disk during encode.
        wav.reset();

        // --- Step 2: semantic slicing --------------------------------------
        std::cout << "[3/5] Semantic slicing (target ~" << opt.target_seg_sec
                  << "s, pause >" << opt.pause_sec << "s, target_clips="
                  << (opt.target_clips > 0 ? std::to_string(opt.target_clips)
                                           : std::string("auto"))
                  << ")...\n";
        SliceOptions slice_opt;
        slice_opt.target_sec = opt.target_seg_sec;
        slice_opt.min_sec = opt.min_seg_sec;
        slice_opt.max_sec = opt.max_seg_sec;
        slice_opt.pause_ms = static_cast<int>(opt.pause_sec * 1000.0);
        slice_opt.target_clips = opt.target_clips;
        slice_opt.verbose = opt.verbose;
        auto segments = slice_segments(words, slice_opt);
        std::cout << "       " << segments.size() << " valid clips\n";
        if (segments.empty()) {
            std::cerr << "[Warn] No segments in [" << opt.min_seg_sec << ","
                      << opt.max_seg_sec << "] s — nothing to render.\n";
            return 0;
        }

        // --- Step 3–5: crop + ASS + render (parallel) ----------------------
        FaceTracker faces;
        faces.load(opt.face_proto, opt.face_model);

        ThreadPool pool(opt.threads > 0
                            ? static_cast<std::size_t>(opt.threads)
                            : 0);
        std::cout << "[4/5] Rendering on " << pool.size() << " workers "
                  << "(encoder=" << encoder_name(opt.encoder) << ")...\n";

        std::atomic<int> done{0};
        const int total = static_cast<int>(segments.size());
        std::vector<std::future<void>> futures;
        futures.reserve(segments.size());

        // Pre-compute crops on the main thread (VideoCapture + DNN sharing is
        // simpler serialized). ASS + FFmpeg encode fan out to the pool.
        for (auto& seg : segments) {
            const CropRect crop =
                faces.compute_crop(input, seg.start_ms, seg.end_ms);

            if (opt.verbose) {
                std::cout << "       clip " << seg.index << ": "
                          << (seg.start_ms / 1000.0) << "s → "
                          << (seg.end_ms / 1000.0) << "s  crop=[" << crop.x
                          << ',' << crop.y << ' ' << crop.w << 'x' << crop.h
                          << "]\n";
            }

            futures.push_back(pool.submit([&, seg, crop]() {
                // Per-job ASS temp — TempPath deletes on return / exception.
                TempPath ass(scratch_dir.path() /
                             ("subs_" + std::to_string(seg.index) + ".ass"));
                write_karaoke_ass(seg, ass.path());

                RenderRequest req;
                req.input_video = input;
                req.output_file =
                    out_dir / ("clip_" + std::to_string(seg.index) + ".mp4");
                req.ass_file = ass.path();
                req.segment = seg;
                req.crop = crop;
                req.encoder = opt.encoder;

                render_clip(req);

                const int n = ++done;
                std::cout << "       [" << n << '/' << total << "] wrote "
                          << req.output_file.filename().string() << '\n';
            }));
        }

        // Propagate the first worker exception (if any).
        std::exception_ptr first_err;
        for (auto& f : futures) {
            try {
                f.get();
            } catch (...) {
                if (!first_err) first_err = std::current_exception();
            }
        }
        if (first_err) std::rethrow_exception(first_err);

        const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - t0)
                            .count();
        std::cout << "[5/5] Done — " << total << " clips in " << (ms / 1000.0)
                  << "s → " << out_dir.string() << '\n';
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "[Fatal] " << e.what() << '\n';
        return 1;
    }
}
