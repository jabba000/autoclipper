#include "whisper_transcriber.hpp"

#include "whisper.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string_view>
#include <thread>

namespace autoclip {

namespace {

/** Map ggml model filename → DTW alignment-head preset (whisper.cpp CLI convention). */
[[nodiscard]] whisper_alignment_heads_preset dtw_preset_for_model(
    const fs::path& model_path) {
    std::string name = model_path.filename().string();
    for (char& c : name) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));

    if (name.find("tiny.en") != std::string::npos) return WHISPER_AHEADS_TINY_EN;
    if (name.find("tiny") != std::string::npos) return WHISPER_AHEADS_TINY;
    if (name.find("base.en") != std::string::npos) return WHISPER_AHEADS_BASE_EN;
    if (name.find("base") != std::string::npos) return WHISPER_AHEADS_BASE;
    if (name.find("small.en") != std::string::npos) return WHISPER_AHEADS_SMALL_EN;
    if (name.find("small") != std::string::npos) return WHISPER_AHEADS_SMALL;
    if (name.find("medium.en") != std::string::npos) return WHISPER_AHEADS_MEDIUM_EN;
    if (name.find("medium") != std::string::npos) return WHISPER_AHEADS_MEDIUM;
    if (name.find("large-v1") != std::string::npos ||
        name.find("large.v1") != std::string::npos)
        return WHISPER_AHEADS_LARGE_V1;
    if (name.find("large-v2") != std::string::npos ||
        name.find("large.v2") != std::string::npos)
        return WHISPER_AHEADS_LARGE_V2;
    if (name.find("turbo") != std::string::npos)
        return WHISPER_AHEADS_LARGE_V3_TURBO;
    if (name.find("large-v3") != std::string::npos ||
        name.find("large.v3") != std::string::npos ||
        name.find("large") != std::string::npos)
        return WHISPER_AHEADS_LARGE_V3;

    // Project default model is ggml-base.bin.
    return WHISPER_AHEADS_BASE;
}

}  // namespace

WhisperContext::WhisperContext(const fs::path& model_path) {
    whisper_context_params cparams = whisper_context_default_params();
    // DTW token alignment — whisper.cpp best practice for word-level timestamps.
    // Incompatible with flash_attn; disable it explicitly.
    cparams.dtw_token_timestamps = true;
    cparams.dtw_aheads_preset = dtw_preset_for_model(model_path);
    cparams.flash_attn = false;

    ctx_ = whisper_init_from_file_with_params(model_path.c_str(), cparams);
    if (!ctx_) {
        throw std::runtime_error("Failed to load Whisper model: " + model_path.string());
    }
}

WhisperContext::~WhisperContext() {
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}

WhisperContext::WhisperContext(WhisperContext&& other) noexcept : ctx_(other.ctx_) {
    other.ctx_ = nullptr;
}

WhisperContext& WhisperContext::operator=(WhisperContext&& other) noexcept {
    if (this != &other) {
        if (ctx_) whisper_free(ctx_);
        ctx_ = other.ctx_;
        other.ctx_ = nullptr;
    }
    return *this;
}

namespace {

struct WavInfo {
    int sample_rate = 0;
    int channels = 0;
    int bits_per_sample = 0;
    std::vector<int16_t> pcm16;
};

uint32_t read_u32_le(const char* p) {
    return static_cast<uint8_t>(p[0]) |
           (static_cast<uint32_t>(static_cast<uint8_t>(p[1])) << 8) |
           (static_cast<uint32_t>(static_cast<uint8_t>(p[2])) << 16) |
           (static_cast<uint32_t>(static_cast<uint8_t>(p[3])) << 24);
}

uint16_t read_u16_le(const char* p) {
    return static_cast<uint8_t>(p[0]) |
           (static_cast<uint16_t>(static_cast<uint8_t>(p[1])) << 8);
}

WavInfo parse_wav(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("Cannot open WAV: " + path.string());

    std::vector<char> file((std::istreambuf_iterator<char>(in)),
                           std::istreambuf_iterator<char>());
    if (file.size() < 44) {
        throw std::runtime_error("WAV too small / corrupt: " + path.string());
    }
    if (std::memcmp(file.data(), "RIFF", 4) != 0 ||
        std::memcmp(file.data() + 8, "WAVE", 4) != 0) {
        throw std::runtime_error("Not a RIFF/WAVE file: " + path.string());
    }

    WavInfo info;
    std::size_t offset = 12;
    const char* data_ptr = nullptr;
    std::size_t data_size = 0;

    while (offset + 8 <= file.size()) {
        const char* chunk_id = file.data() + offset;
        uint32_t chunk_sz = read_u32_le(file.data() + offset + 4);
        offset += 8;
        if (offset + chunk_sz > file.size()) {
            throw std::runtime_error("Truncated WAV chunk in " + path.string());
        }

        if (std::memcmp(chunk_id, "fmt ", 4) == 0) {
            if (chunk_sz < 16) throw std::runtime_error("Invalid fmt chunk");
            const char* fmt = file.data() + offset;
            uint16_t audio_fmt = read_u16_le(fmt);
            info.channels = read_u16_le(fmt + 2);
            info.sample_rate = static_cast<int>(read_u32_le(fmt + 4));
            info.bits_per_sample = read_u16_le(fmt + 14);
            if (audio_fmt != 1) {
                throw std::runtime_error("Only PCM WAV is supported");
            }
        } else if (std::memcmp(chunk_id, "data", 4) == 0) {
            data_ptr = file.data() + offset;
            data_size = chunk_sz;
        }
        offset += chunk_sz + (chunk_sz & 1u);
    }

    if (!data_ptr || info.channels < 1 || info.bits_per_sample != 16) {
        throw std::runtime_error("WAV missing PCM16 data chunk: " + path.string());
    }

    const std::size_t n_samples = data_size / sizeof(int16_t);
    info.pcm16.resize(n_samples);
    std::memcpy(info.pcm16.data(), data_ptr, n_samples * sizeof(int16_t));
    return info;
}

// ---------------------------------------------------------------------------
// UTF-8 / token helpers for Ukrainian BPE merge
// ---------------------------------------------------------------------------

[[nodiscard]] bool is_ascii_space(unsigned char c) noexcept {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

/** Strip leading/trailing ASCII whitespace (Whisper BPE uses ASCII space). */
[[nodiscard]] std::string_view trim_ascii(std::string_view t) noexcept {
    while (!t.empty() && is_ascii_space(static_cast<unsigned char>(t.front()))) {
        t.remove_prefix(1);
    }
    while (!t.empty() && is_ascii_space(static_cast<unsigned char>(t.back()))) {
        t.remove_suffix(1);
    }
    return t;
}

[[nodiscard]] bool is_control_token_text(std::string_view t) noexcept {
    if (t.empty()) return true;
    // [_BEG_], [BLANK_AUDIO], <|startoftranscript|>, etc.
    if (t.front() == '[' || t.front() == '<') return true;
    if (trim_ascii(t).empty()) return true;
    return false;
}

/** Ukrainian / Latin apostrophe codepoints that never start a new word. */
[[nodiscard]] bool starts_with_apostrophe(std::string_view t) noexcept {
    if (t.empty()) return false;
    const auto* p = reinterpret_cast<const unsigned char*>(t.data());
    // ASCII ' or `
    if (p[0] == '\'' || p[0] == '`') return true;
    // U+2019 RIGHT SINGLE QUOTATION MARK  E2 80 99
    if (t.size() >= 3 && p[0] == 0xE2 && p[1] == 0x80 && p[2] == 0x99) return true;
    // U+02BC MODIFIER LETTER APOSTROPHE  CA BC
    if (t.size() >= 2 && p[0] == 0xCA && p[1] == 0xBC) return true;
    // U+2018 LEFT SINGLE QUOTATION MARK  E2 80 98
    if (t.size() >= 3 && p[0] == 0xE2 && p[1] == 0x80 && p[2] == 0x98) return true;
    return false;
}

/**
 * True if every UTF-8 codepoint in t is punctuation / symbol (no letters/digits).
 * Used to glue trailing ".", ",", "!" onto the preceding word.
 */
[[nodiscard]] bool is_punctuation_only(std::string_view t) noexcept {
    t = trim_ascii(t);
    if (t.empty()) return true;

    auto is_letter_or_digit = [](uint32_t cp) -> bool {
        if (cp >= '0' && cp <= '9') return true;
        if (cp >= 'A' && cp <= 'Z') return true;
        if (cp >= 'a' && cp <= 'z') return true;
        // Cyrillic block (covers Ukrainian)
        if (cp >= 0x0400 && cp <= 0x04FF) return true;
        // Ukrainian i / ï sometimes outside? covered in Cyrillic.
        // Latin supplements rarely needed for UK speech tokens.
        if (cp >= 0x00C0 && cp <= 0x024F) return true;
        return false;
    };

    std::size_t i = 0;
    bool any = false;
    while (i < t.size()) {
        const auto c = static_cast<unsigned char>(t[i]);
        uint32_t cp = 0;
        int len = 1;
        if ((c & 0x80) == 0) {
            cp = c;
            len = 1;
        } else if ((c & 0xE0) == 0xC0 && i + 1 < t.size()) {
            cp = (c & 0x1Fu) << 6 | (static_cast<unsigned char>(t[i + 1]) & 0x3Fu);
            len = 2;
        } else if ((c & 0xF0) == 0xE0 && i + 2 < t.size()) {
            cp = (c & 0x0Fu) << 12 |
                 (static_cast<unsigned char>(t[i + 1]) & 0x3Fu) << 6 |
                 (static_cast<unsigned char>(t[i + 2]) & 0x3Fu);
            len = 3;
        } else if ((c & 0xF8) == 0xF0 && i + 3 < t.size()) {
            cp = (c & 0x07u) << 18 |
                 (static_cast<unsigned char>(t[i + 1]) & 0x3Fu) << 12 |
                 (static_cast<unsigned char>(t[i + 2]) & 0x3Fu) << 6 |
                 (static_cast<unsigned char>(t[i + 3]) & 0x3Fu);
            len = 4;
        } else {
            return false;  // malformed — treat as content
        }
        if (is_letter_or_digit(cp)) return false;
        any = true;
        i += static_cast<std::size_t>(len);
    }
    return any;
}

struct RawTok {
    std::string text;   // raw whisper token text (may include leading space)
    int64_t t0_ms = 0;
    int64_t t1_ms = 0;
};

/**
 * Merge Whisper BPE tokens into linguistic words.
 *
 * Word boundary rule (whisper.cpp / GPT-2 BPE):
 *   - A token whose text begins with ASCII space starts a new word.
 *   - Tokens without a leading space continue the current word (syllable/subword).
 *   - Apostrophe-leading continuations always glue to the previous word.
 *   - Punctuation-only tokens glue to the previous word (even with a leading space).
 *
 * Timestamps: word start = first token t0; word end = last token t1.
 */
[[nodiscard]] std::vector<Word> merge_tokens_to_words(const std::vector<RawTok>& tokens) {
    std::vector<Word> words;
    words.reserve(tokens.size() / 2 + 1);

    auto flush_ready = [&](Word& w) {
        // Collapse internal spaces introduced by bad merges; keep as single display word.
        std::string cleaned;
        cleaned.reserve(w.text.size());
        bool prev_space = false;
        for (char ch : w.text) {
            if (is_ascii_space(static_cast<unsigned char>(ch))) {
                if (!cleaned.empty() && !prev_space) {
                    cleaned.push_back(' ');
                    prev_space = true;
                }
            } else {
                cleaned.push_back(ch);
                prev_space = false;
            }
        }
        while (!cleaned.empty() && is_ascii_space(static_cast<unsigned char>(cleaned.back()))) {
            cleaned.pop_back();
        }
        w.text = std::move(cleaned);
        if (!w.text.empty()) {
            words.push_back(std::move(w));
        }
    };

    Word cur;
    bool open = false;

    for (const auto& tok : tokens) {
        std::string_view raw = tok.text;
        if (raw.empty()) continue;

        const bool lead_space = is_ascii_space(static_cast<unsigned char>(raw.front()));
        std::string_view body = trim_ascii(raw);
        if (body.empty()) continue;

        const bool punct = is_punctuation_only(body);
        const bool apo = starts_with_apostrophe(body);

        // Glue punctuation / apostrophe fragments onto the open word.
        if (open && (punct || apo || !lead_space)) {
            cur.text.append(body);
            cur.t1_ms = tok.t1_ms;
            continue;
        }

        // Leading space (or first token) → new word. Punctuation with no open word: skip.
        if (punct && !open) {
            continue;
        }

        if (open) {
            flush_ready(cur);
            cur = Word{};
            open = false;
        }

        cur.text = std::string(body);
        cur.t0_ms = tok.t0_ms;
        cur.t1_ms = tok.t1_ms;
        open = true;
    }

    if (open) {
        flush_ready(cur);
    }

    // Drop degenerate single-byte junk that slipped through (should be rare).
    words.erase(std::remove_if(words.begin(), words.end(),
                               [](const Word& w) {
                                   return w.text.empty() || w.t1_ms < w.t0_ms;
                               }),
                words.end());

    return words;
}

/**
 * Clamp pathological token-timestamp spans (common on base models at
 * segment edges) so subtitle chunks follow speech rhythm. Does not invent
 * overlapping times: never extends past the next word's start.
 */
void refine_word_durations(std::vector<Word>& words) {
    auto codepoints = [](std::string_view s) {
        std::size_t n = 0;
        for (unsigned char c : s) {
            if ((c & 0xC0) != 0x80) ++n;
        }
        return std::max<std::size_t>(1, n);
    };

    for (std::size_t i = 0; i < words.size(); ++i) {
        const std::size_t cps = codepoints(trim_ascii(words[i].text));
        const int64_t max_ms =
            std::max<int64_t>(900, static_cast<int64_t>(cps) * 180);
        const int64_t min_ms =
            std::max<int64_t>(60, static_cast<int64_t>(cps) * 35);

        int64_t t0 = words[i].t0_ms;
        int64_t t1 = words[i].t1_ms;
        const int64_t orig_t1 = t1;
        if (i + 1 < words.size()) {
            t1 = std::min(t1, words[i + 1].t0_ms);
        }

        const int64_t dur = t1 - t0;
        if (dur > max_ms) {
            t1 = t0 + max_ms;
            // Preserve adjacency: don't invent a pause that splits glued speech
            // ("І" + "так,") into separate subtitle chunks.
            if (i + 1 < words.size() &&
                words[i + 1].t0_ms <= orig_t1 + 50) {
                words[i + 1].t0_ms = t1;
            }
        } else if (dur < min_ms) {
            int64_t want = t0 + min_ms;
            if (i + 1 < words.size()) {
                want = std::min(want, words[i + 1].t0_ms);
            }
            t1 = std::max(t1, want);
        }
        if (t1 <= t0) t1 = t0 + 60;

        words[i].t0_ms = t0;
        words[i].t1_ms = t1;
    }
}

}  // namespace

std::vector<float> load_wav_f32(const fs::path& wav_path) {
    WavInfo info = parse_wav(wav_path);

    const int ch = info.channels;
    const std::size_t frames = info.pcm16.size() / static_cast<std::size_t>(ch);
    std::vector<float> out(frames);
    for (std::size_t i = 0; i < frames; ++i) {
        float sum = 0.f;
        for (int c = 0; c < ch; ++c) {
            sum += static_cast<float>(info.pcm16[i * static_cast<std::size_t>(ch) +
                                                 static_cast<std::size_t>(c)]);
        }
        out[i] = (sum / static_cast<float>(ch)) / 32768.0f;
    }

    if (info.sample_rate != WHISPER_SAMPLE_RATE) {
        throw std::runtime_error(
            "WAV sample rate is " + std::to_string(info.sample_rate) +
            " Hz; expected " + std::to_string(WHISPER_SAMPLE_RATE) +
            " Hz. Re-extract audio with FFmpeg -ar 16000.");
    }
    return out;
}

std::vector<Word> transcribe(WhisperContext& ctx, const fs::path& wav_path,
                             const std::string& language, int n_threads) {
    auto pcmf32 = load_wav_f32(wav_path);

    // Beam search is the OpenAI / whisper.cpp quality default (beam_size=5).
    // Cost is paid once during transcription; render pool is unaffected.
    whisper_full_params wparams =
        whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH);

    wparams.language = language.c_str();
    wparams.translate = false;
    wparams.detect_language = false;

    // Accurate word timestamps without fragmenting the decoder output.
    // max_len=1 was forcing per-token segment wraps and destroying words.
    wparams.token_timestamps = true;
    wparams.max_len = 0;          // no character wrap
    wparams.split_on_word = false;  // irrelevant when max_len==0; keep off
    wparams.max_tokens = 0;       // no artificial token cap per segment

    wparams.no_context = false;   // carry prompt context across windows
    wparams.single_segment = false;
    wparams.print_special = false;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;

    // Suppress blanks / non-speech for cleaner Ukrainian transcripts.
    wparams.suppress_blank = true;
    wparams.suppress_nst = true;

    // Token-timestamp thresholds: whisper.cpp defaults (~0.01) are fine;
    // leave thold_pt / thold_ptsum from whisper_full_default_params.

    wparams.beam_search.beam_size = 5;
    wparams.temperature = 0.0f;

    wparams.n_threads =
        n_threads > 0 ? n_threads
                      : static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));

    if (whisper_full(ctx.get(), wparams, pcmf32.data(),
                     static_cast<int>(pcmf32.size())) != 0) {
        throw std::runtime_error("whisper_full inference failed");
    }

    const whisper_token eot = whisper_token_eot(ctx.get());
    std::vector<RawTok> raw;
    const int n_segments = whisper_full_n_segments(ctx.get());
    raw.reserve(static_cast<std::size_t>(n_segments) * 16);

    for (int i = 0; i < n_segments; ++i) {
        const int n_tokens = whisper_full_n_tokens(ctx.get(), i);
        for (int j = 0; j < n_tokens; ++j) {
            const whisper_token id = whisper_full_get_token_id(ctx.get(), i, j);
            // Skip special / timestamp control tokens (id >= EOT).
            if (id >= eot) continue;

            const char* raw_txt = whisper_full_get_token_text(ctx.get(), i, j);
            if (!raw_txt) continue;
            if (is_control_token_text(raw_txt)) continue;

            const whisper_token_data td =
                whisper_full_get_token_data(ctx.get(), i, j);

            // t0/t1 are centiseconds → milliseconds.
            raw.push_back(RawTok{
                std::string(raw_txt),
                static_cast<int64_t>(td.t0) * 10,
                static_cast<int64_t>(td.t1) * 10,
            });
        }
    }

    auto words = merge_tokens_to_words(raw);
    refine_word_durations(words);
    return words;
}

}  // namespace autoclip
