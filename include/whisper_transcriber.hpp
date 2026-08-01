#pragma once

/**
 * Whisper transcription (Ukrainian word-level timestamps)
 * -------------------------------------------------------
 * Loads a ggml model once via whisper_init_from_file_with_params and runs
 * whisper_full with token_timestamps=true.
 *
 * Best-practice params (see transcribe()):
 *   - context: dtw_token_timestamps + model-matched alignment heads
 *   - beam search (beam_size=5) for higher Ukrainian accuracy
 *   - max_len=0  — do NOT fragment segments (max_len=1 was splitting words)
 *   - split_on_word unused when max_len==0
 *   - language pinned (default "uk"); no auto-detect noise
 *
 * Post-process: BPE tokens are merged into linguistic Word objects.
 * Whisper marks word boundaries with a leading ASCII space on the first
 * token of each word. Continuation tokens (no leading space) are appended.
 * Word.t0_ms = first token start; Word.t1_ms = last token end.
 *
 * The WhisperContext RAII wrapper frees the C context on destruction.
 */

#include "types.hpp"

#include <filesystem>
#include <string>
#include <vector>

struct whisper_context;

namespace autoclip {

namespace fs = std::filesystem;

class WhisperContext {
public:
    explicit WhisperContext(const fs::path& model_path);
    ~WhisperContext();

    WhisperContext(const WhisperContext&) = delete;
    WhisperContext& operator=(const WhisperContext&) = delete;
    WhisperContext(WhisperContext&&) noexcept;
    WhisperContext& operator=(WhisperContext&&) noexcept;

    [[nodiscard]] whisper_context* get() const noexcept { return ctx_; }

private:
    whisper_context* ctx_ = nullptr;
};

/** Load 16-bit PCM mono WAV into float32 samples in [-1, 1]. Handles RIFF chunks. */
[[nodiscard]] std::vector<float> load_wav_f32(const fs::path& wav_path);

/**
 * Transcribe WAV and return merged linguistic words with timestamps.
 * @param language ISO-639-1 code (default "uk").
 */
[[nodiscard]] std::vector<Word> transcribe(
    WhisperContext& ctx,
    const fs::path& wav_path,
    const std::string& language = "uk",
    int n_threads = 0);

}  // namespace autoclip
