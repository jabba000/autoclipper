#include "ass_generator.hpp"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace autoclip {

namespace {

std::string ass_time(int64_t ms) {
    if (ms < 0) ms = 0;
    const int h = static_cast<int>(ms / 3'600'000);
    const int m = static_cast<int>((ms % 3'600'000) / 60'000);
    const int s = static_cast<int>((ms % 60'000) / 1000);
    const int cs = static_cast<int>((ms % 1000) / 10);
    std::ostringstream oss;
    oss << h << ':' << std::setfill('0') << std::setw(2) << m << ':'
        << std::setw(2) << s << '.' << std::setw(2) << cs;
    return oss.str();
}

std::string escape_ass_text(const std::string& in) {
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        if (c == '{' || c == '}') continue;
        if (c == '\n' || c == '\r') {
            out += "\\N";
            continue;
        }
        out.push_back(c);
    }
    return out;
}

/** UTF-8 codepoint count (not grapheme clusters — enough for letter checks). */
[[nodiscard]] std::size_t utf8_codepoints(std::string_view s) noexcept {
    std::size_t n = 0;
    for (unsigned char c : s) {
        if ((c & 0xC0) != 0x80) ++n;
    }
    return n;
}

/** Strip trailing ASCII punctuation for length checks. */
[[nodiscard]] std::string_view strip_trailing_punct(std::string_view s) noexcept {
    auto is_punct = [](unsigned char c) {
        return c == '.' || c == ',' || c == '!' || c == '?' || c == ':' ||
               c == ';' || c == '"' || c == '\'' || c == ')' || c == ']';
    };
    while (!s.empty() && is_punct(static_cast<unsigned char>(s.back()))) {
        s.remove_suffix(1);
    }
    return s;
}

[[nodiscard]] bool is_single_letter_word(const Word& w) noexcept {
    const auto core = strip_trailing_punct(w.text);
    return utf8_codepoints(core) <= 1;
}

/**
 * OpusClip chunking:
 *   - prefer 2 words, hard max 3
 *   - break on long inter-word pauses (new subtitle)
 *   - never emit a lone single-letter chunk when avoidable
 *   - never leave a single-letter orphan after a 2-word take if a 3rd fits
 */
[[nodiscard]] std::size_t choose_chunk_size(const std::vector<Word>& words,
                                            std::size_t i,
                                            int64_t pause_ms) {
    const std::size_t remain = words.size() - i;
    if (remain == 0) return 0;
    if (remain == 1) return 1;

    auto gap_before = [&](std::size_t idx) -> int64_t {
        if (idx == 0 || idx >= words.size()) return 0;
        return words[idx].t0_ms - words[idx - 1].t1_ms;
    };

    // Prefer 2; allow 3 when it prevents orphans / fits natural rhythm.
    std::size_t n = std::min<std::size_t>(2, remain);

    const bool lead_tiny = is_single_letter_word(words[i]);
    const int64_t gap1 = gap_before(i + 1);

    // Long pause → new subtitle. Never flash a lone single-letter word:
    // glue it to the next word across gaps under 1.5s.
    if (remain >= 2 && gap1 >= pause_ms) {
        if (!(lead_tiny && gap1 < 1500)) {
            return 1;
        }
    }

    // Lead single-letter / preposition: always pair with the next word.
    if (lead_tiny && remain >= 2 && gap1 < 1500) {
        n = 2;
        if (remain >= 3 && is_single_letter_word(words[i + 2]) &&
            gap_before(i + 2) < pause_ms) {
            n = 3;
        }
        return n;
    }

    // Pull a 3rd word when it prevents a single-letter orphan or speech is tight.
    if (remain >= 3 && gap_before(i + 2) < pause_ms) {
        const bool third_is_tiny = is_single_letter_word(words[i + 2]);
        const bool would_orphan_tiny =
            (remain == 3 && is_single_letter_word(words[i + 2]));
        const bool tight_triplet =
            (words[i + 2].t0_ms - words[i].t0_ms) <= 1800;

        if (would_orphan_tiny || third_is_tiny || (remain == 3 && tight_triplet)) {
            n = 3;
        }
    }

    if (n == 2 && remain == 3 && is_single_letter_word(words[i + 2]) &&
        gap_before(i + 2) < pause_ms) {
        n = 3;
    }

    return std::min(n, remain);
}

}  // namespace

fs::path write_karaoke_ass(const Segment& seg, const fs::path& out_path) {
    std::ofstream ass(out_path);
    if (!ass) {
        throw std::runtime_error("Cannot write ASS: " + out_path.string());
    }

    // PlayRes matches final 9:16 framing.
    ass << "[Script Info]\n"
           "Title: AutoClipper Karaoke\n"
           "ScriptType: v4.00+\n"
           "WrapStyle: 0\n"
           "ScaledBorderAndShadow: yes\n"
           "YCbCr Matrix: None\n"
           "PlayResX: 1080\n"
           "PlayResY: 1920\n\n";

    // TikTok style, lower-third placement:
    //   Alignment 2 = bottom-center.
    //   MarginV 480 → text sits ~70–75% down (1920-480=1440 ≈ 75% line),
    //   keeping faces in the upper/mid frame clear.
    // PrimaryColour yellow = karaoke "spoken" fill;
    // SecondaryColour white = "upcoming" fill (standard ASS \\k behaviour).
    // No Shadow — reduces shimmer; thick Outline for readability.
    ass << "[V4+ Styles]\n"
           "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
           "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
           "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
           "Alignment, MarginL, MarginR, MarginV, Encoding\n"
           "Style: TikTok,Arial Black,92,&H0000FFFF,&H00FFFFFF,&H00000000,"
           "&H80000000,-1,0,0,0,100,100,0,0,1,7,0,2,60,60,480,1\n\n";

    ass << "[Events]\n"
           "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
           "Effect, Text\n";

    const int64_t base = seg.start_ms;
    constexpr int64_t kPauseMs = 400;     // long pause → new subtitle
    constexpr int64_t kMinChunkMs = 350;  // floor to avoid rapid flashing

    int64_t prev_end_local = -1;  // ensure non-overlapping, stable events

    for (std::size_t i = 0; i < seg.words.size();) {
        const std::size_t n = choose_chunk_size(seg.words, i, kPauseMs);
        if (n == 0) break;

        const auto& first = seg.words[i];
        const auto& last = seg.words[i + n - 1];

        int64_t t0 = std::max<int64_t>(0, first.t0_ms - base);
        int64_t t1 = std::max<int64_t>(t0 + kMinChunkMs, last.t1_ms - base);

        // Hold the full chunk on screen until it has finished speaking.
        // If the next event would start earlier, clamp so events never overlap
        // (overlap causes ASS flicker / z-fight).
        if (prev_end_local >= 0 && t0 < prev_end_local) {
            t0 = prev_end_local;
        }
        if (t1 <= t0) t1 = t0 + kMinChunkMs;

        // Extend hold slightly into a following pause (readability), but never
        // into the next word's start.
        if (i + n < seg.words.size()) {
            const int64_t next_local =
                std::max<int64_t>(0, seg.words[i + n].t0_ms - base);
            const int64_t hold = std::min(next_local, t1 + 80);
            if (hold > t1) t1 = hold;
        }

        ass << "Dialogue: 0," << ass_time(t0) << ',' << ass_time(t1)
            << ",TikTok,,0,0,0,,";

        // Single Dialogue event for the whole chunk — subtitle text is stable.
        // \\k durations sum approximately to (t1-t0); karaoke recolours words
        // in place (Primary←Secondary) without spawning new events.
        int64_t k_budget_cs = std::max<int64_t>(1, (t1 - t0) / 10);
        int64_t k_assigned = 0;

        for (std::size_t j = 0; j < n; ++j) {
            const auto& w = seg.words[i + j];
            int64_t dur_cs = (w.t1_ms - w.t0_ms) / 10;
            if (dur_cs < 1) dur_cs = 1;

            // Last word absorbs remaining karaoke budget so \\k fills the event.
            if (j + 1 == n) {
                dur_cs = std::max<int64_t>(1, k_budget_cs - k_assigned);
            }
            k_assigned += dur_cs;

            // Pure \\k — do not inject \\1c/\\2c overrides (they fight the
            // karaoke fill and cause colour flicker under libass).
            ass << "{\\k" << dur_cs << '}' << escape_ass_text(w.text);
            if (j + 1 < n) ass << ' ';
        }
        ass << '\n';

        prev_end_local = t1;
        i += n;
    }

    ass.flush();
    if (!ass) {
        throw std::runtime_error("Failed writing ASS: " + out_path.string());
    }
    return out_path;
}

}  // namespace autoclip
