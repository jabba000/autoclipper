#include "segmenter.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace autoclip {

namespace {

struct WordSpan {
    std::size_t begin = 0;  // inclusive index into words
    std::size_t end = 0;    // exclusive
};

struct Candidate {
    WordSpan span{};
    int64_t start_ms = 0;
    int64_t end_ms = 0;
    double score = 0.0;
    double density = 0.0;
    bool ends_sentence = false;
};

[[nodiscard]] int64_t span_start(const std::vector<Word>& w, WordSpan s) {
    return w[s.begin].t0_ms;
}

[[nodiscard]] int64_t span_end(const std::vector<Word>& w, WordSpan s) {
    return w[s.end - 1].t1_ms;
}

[[nodiscard]] int64_t span_dur(const std::vector<Word>& w, WordSpan s) {
    return span_end(w, s) - span_start(w, s);
}

[[nodiscard]] bool ends_with_sentence_punct(std::string_view text) {
    // Strip trailing quotes / brackets then check terminal punctuation.
    while (!text.empty()) {
        const unsigned char c = static_cast<unsigned char>(text.back());
        if (c == '"' || c == '\'' || c == ')' || c == ']') {
            text.remove_suffix(1);
            continue;
        }
        // UTF-8 » (U+00BB) = C2 BB
        if (text.size() >= 2) {
            const auto* p = reinterpret_cast<const unsigned char*>(
                text.data() + text.size() - 2);
            if (p[0] == 0xC2 && p[1] == 0xBB) {
                text.remove_suffix(2);
                continue;
            }
        }
        break;
    }
    if (text.empty()) return false;

    const unsigned char last = static_cast<unsigned char>(text.back());
    if (last == '.' || last == '!' || last == '?') return true;

    // UTF-8 ellipsis U+2026 = E2 80 A6
    if (text.size() >= 3) {
        const auto* p = reinterpret_cast<const unsigned char*>(text.data() +
                                                               text.size() - 3);
        if (p[0] == 0xE2 && p[1] == 0x80 && p[2] == 0xA6) return true;
    }
    // ASCII "..."
    if (text.size() >= 3 && text.substr(text.size() - 3) == "...") return true;
    return false;
}

[[nodiscard]] double speech_density(const std::vector<Word>& words,
                                    WordSpan s) {
    const int64_t wall = span_dur(words, s);
    if (wall <= 0) return 0.0;
    int64_t spoken = 0;
    for (std::size_t i = s.begin; i < s.end; ++i) {
        spoken += std::max<int64_t>(0, words[i].t1_ms - words[i].t0_ms);
    }
    return static_cast<double>(spoken) / static_cast<double>(wall);
}

[[nodiscard]] std::string fmt_range(int64_t a_ms, int64_t b_ms) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.1fs–%.1fs", a_ms / 1000.0,
                  b_ms / 1000.0);
    return buf;
}

void log_rejected(bool verbose, int64_t a, int64_t b, const char* why) {
    if (!verbose) return;
    std::cout << "Rejected: " << fmt_range(a, b) << "  " << why << '\n';
}

void log_accepted(bool verbose, int64_t a, int64_t b, double score) {
    if (!verbose) return;
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.2f", score);
    std::cout << "Accepted: " << fmt_range(a, b) << "  score=" << buf << '\n';
}

/** Phase 1: always split on pauses — never bridge long silence. */
[[nodiscard]] std::vector<WordSpan> build_speech_regions(
    const std::vector<Word>& words, int pause_ms) {
    std::vector<WordSpan> regions;
    if (words.empty()) return regions;

    const int64_t hard_gap_ms = 2000;  // always split at ≥2s even if pause_sec is higher

    WordSpan cur{0, 1};
    for (std::size_t i = 1; i < words.size(); ++i) {
        const int64_t gap = words[i].t0_ms - words[i - 1].t1_ms;
        if (gap >= pause_ms || gap >= hard_gap_ms) {
            cur.end = i;
            regions.push_back(cur);
            cur = WordSpan{i, i + 1};
        } else {
            cur.end = i + 1;
        }
    }
    cur.end = words.size();
    regions.push_back(cur);
    return regions;
}

/**
 * Best cut index inside (begin, end) after at least min_pack words worth of
 * time from pack_start. Returns index of last word IN the left piece
 * (exclusive end = cut+1), or npos if none.
 */
[[nodiscard]] std::size_t find_best_cut(const std::vector<Word>& words,
                                        std::size_t begin, std::size_t end,
                                        std::size_t pack_start,
                                        int64_t min_pack_ms, int64_t ideal_ms,
                                        int64_t max_ms, int pause_ms) {
    if (end <= begin + 1) return static_cast<std::size_t>(-1);

    const int64_t t0 = words[pack_start].t0_ms;
    std::size_t best_pause = static_cast<std::size_t>(-1);
    std::size_t best_sent = static_cast<std::size_t>(-1);
    std::size_t best_dur = static_cast<std::size_t>(-1);
    int64_t best_pause_dist = INT64_MAX;
    int64_t best_sent_dist = INT64_MAX;
    int64_t best_dur_dist = INT64_MAX;

    for (std::size_t i = pack_start; i + 1 < end; ++i) {
        const int64_t dur = words[i].t1_ms - t0;
        if (dur < min_pack_ms) continue;
        if (dur > max_ms) break;

        const int64_t gap = words[i + 1].t0_ms - words[i].t1_ms;
        const int64_t dist = std::llabs(dur - ideal_ms);

        if (gap >= pause_ms && dist < best_pause_dist) {
            best_pause_dist = dist;
            best_pause = i;
        }
        if (ends_with_sentence_punct(words[i].text) && dist < best_sent_dist) {
            best_sent_dist = dist;
            best_sent = i;
        }
        if (dur >= ideal_ms && dist < best_dur_dist) {
            best_dur_dist = dist;
            best_dur = i;
        }
    }

    // Priority: pause → sentence → duration. Also force at max.
    if (best_pause != static_cast<std::size_t>(-1)) return best_pause;
    if (best_sent != static_cast<std::size_t>(-1)) return best_sent;
    if (best_dur != static_cast<std::size_t>(-1)) return best_dur;

    // Hard duration stop: last word before exceeding max_ms.
    for (std::size_t i = pack_start; i + 1 < end; ++i) {
        const int64_t dur = words[i].t1_ms - t0;
        const int64_t next_dur = words[i + 1].t1_ms - t0;
        if (dur >= min_pack_ms && next_dur > max_ms) return i;
    }

    return static_cast<std::size_t>(-1);
}

/** Phase 2: subdivide one speech region into raw candidate spans. */
[[nodiscard]] std::vector<WordSpan> cut_region(const std::vector<Word>& words,
                                               WordSpan region,
                                               const SliceOptions& opt) {
    std::vector<WordSpan> out;
    const int64_t target_ms = static_cast<int64_t>(opt.target_sec) * 1000;
    const int64_t min_ms = static_cast<int64_t>(opt.min_sec) * 1000;
    const int64_t max_ms = static_cast<int64_t>(opt.max_sec) * 1000;
    // Prefer packing into 15–30s when forcing; floor for sentence cuts.
    const int64_t pack_floor_ms =
        std::min(target_ms, std::max<int64_t>(15'000, target_ms / 2));
    const int64_t sent_floor_ms = 15'000;

    const int64_t region_dur = span_dur(words, region);
    if (region_dur < min_ms) {
        log_rejected(opt.verbose, span_start(words, region),
                     span_end(words, region), "duration too short");
        return out;
    }

    // Short enough: single candidate.
    if (region_dur <= target_ms) {
        out.push_back(region);
        return out;
    }

    std::size_t cursor = region.begin;
    while (cursor < region.end) {
        const int64_t remain =
            words[region.end - 1].t1_ms - words[cursor].t0_ms;
        if (remain <= target_ms) {
            out.push_back(WordSpan{cursor, region.end});
            break;
        }

        // Prefer sentence cuts from 15s; pause/duration from pack_floor.
        std::size_t cut = find_best_cut(words, region.begin, region.end, cursor,
                                        pack_floor_ms, target_ms, max_ms,
                                        opt.pause_ms);

        // Retry with sentence floor if pause-based search used higher floor
        // but found nothing — find_best_cut already considers sentences from
        // min_pack_ms; try a lower floor for sentence-only path.
        if (cut == static_cast<std::size_t>(-1)) {
            cut = find_best_cut(words, region.begin, region.end, cursor,
                                sent_floor_ms, target_ms, max_ms, opt.pause_ms);
        }

        if (cut == static_cast<std::size_t>(-1) || cut + 1 >= region.end) {
            // Cannot cut further — emit remainder (may be filtered later).
            out.push_back(WordSpan{cursor, region.end});
            break;
        }

        out.push_back(WordSpan{cursor, cut + 1});
        cursor = cut + 1;
    }

    return out;
}

[[nodiscard]] double score_candidate(const Candidate& c, int64_t target_ms) {
    const int64_t dur = c.end_ms - c.start_ms;
    const double dur_score =
        1.0 - std::min(1.0, std::fabs(static_cast<double>(dur - target_ms)) /
                                static_cast<double>(target_ms));
    const double dens = c.density;
    const double sent = c.ends_sentence ? 0.15 : 0.0;
    return 0.45 * dur_score + 0.40 * dens + sent;
}

[[nodiscard]] Candidate make_candidate(const std::vector<Word>& words,
                                       WordSpan span, int64_t target_ms) {
    Candidate c;
    c.span = span;
    c.start_ms = span_start(words, span);
    c.end_ms = span_end(words, span);
    c.density = speech_density(words, span);
    c.ends_sentence = ends_with_sentence_punct(words[span.end - 1].text);
    c.score = score_candidate(c, target_ms);
    return c;
}

[[nodiscard]] bool filter_candidate(const Candidate& c, const SliceOptions& opt,
                                    bool log) {
    const int64_t min_ms = static_cast<int64_t>(opt.min_sec) * 1000;
    const int64_t max_ms = static_cast<int64_t>(opt.max_sec) * 1000;
    const int64_t dur = c.end_ms - c.start_ms;

    if (dur < min_ms) {
        log_rejected(log, c.start_ms, c.end_ms, "duration too short");
        return false;
    }
    if (dur > max_ms) {
        log_rejected(log, c.start_ms, c.end_ms, "outside duration limits");
        return false;
    }
    constexpr double kMinDensity = 0.55;
    if (c.density < kMinDensity) {
        log_rejected(log, c.start_ms, c.end_ms, "too much silence");
        return false;
    }
    return true;
}

[[nodiscard]] int resolve_target_clips(const std::vector<Word>& words,
                                       const SliceOptions& opt) {
    if (opt.target_clips > 0) return opt.target_clips;
    if (words.empty()) return 5;
    const double timeline_sec =
        (words.back().t1_ms - words.front().t0_ms) / 1000.0;
    const int raw = static_cast<int>(
        std::llround(timeline_sec / std::max(1, opt.target_sec)));
    return std::clamp(raw, 5, 12);
}

/**
 * Split one candidate at its best internal cut into two legal pieces.
 * Returns false if no legal split exists.
 */
bool try_split_candidate(const std::vector<Word>& words, Candidate& src,
                         std::vector<Candidate>& dst, const SliceOptions& opt) {
    const int64_t target_ms = static_cast<int64_t>(opt.target_sec) * 1000;
    const int64_t min_ms = static_cast<int64_t>(opt.min_sec) * 1000;
    const int64_t max_ms = static_cast<int64_t>(opt.max_sec) * 1000;
    const int64_t pack_floor = std::max(min_ms, target_ms / 2);

    if (src.span.end <= src.span.begin + 1) return false;

    std::size_t cut =
        find_best_cut(words, src.span.begin, src.span.end, src.span.begin,
                      pack_floor, target_ms, max_ms, opt.pause_ms);
    if (cut == static_cast<std::size_t>(-1)) {
        // Midpoint word split as last resort.
        const std::size_t mid = (src.span.begin + src.span.end) / 2;
        if (mid <= src.span.begin || mid >= src.span.end) return false;
        cut = mid - 1;
    }

    WordSpan left{src.span.begin, cut + 1};
    WordSpan right{cut + 1, src.span.end};
    if (left.end <= left.begin || right.end <= right.begin) return false;

    Candidate a = make_candidate(words, left, target_ms);
    Candidate b = make_candidate(words, right, target_ms);
    if (!filter_candidate(a, opt, /*log=*/false) ||
        !filter_candidate(b, opt, /*log=*/false)) {
        return false;
    }

    dst.push_back(a);
    dst.push_back(b);
    return true;
}

[[nodiscard]] std::vector<Candidate> select_diverse(
    std::vector<Candidate> cands, int target_n) {
    if (static_cast<int>(cands.size()) <= target_n) return cands;
    if (target_n <= 0) return {};

    std::sort(cands.begin(), cands.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.start_ms < b.start_ms;
              });

    const int64_t t0 = cands.front().start_ms;
    const int64_t t1 = cands.back().end_ms;
    const int64_t span = std::max<int64_t>(1, t1 - t0);

    std::vector<bool> taken(cands.size(), false);
    std::vector<Candidate> picked;
    picked.reserve(static_cast<std::size_t>(target_n));

    // One best pick per timeline bin.
    for (int bin = 0; bin < target_n; ++bin) {
        const int64_t bin_lo = t0 + (span * bin) / target_n;
        const int64_t bin_hi = t0 + (span * (bin + 1)) / target_n;
        int best = -1;
        for (std::size_t i = 0; i < cands.size(); ++i) {
            if (taken[i]) continue;
            const int64_t mid = (cands[i].start_ms + cands[i].end_ms) / 2;
            if (mid < bin_lo || mid >= bin_hi) continue;
            if (best < 0 || cands[i].score > cands[static_cast<std::size_t>(best)].score)
                best = static_cast<int>(i);
        }
        // Fallback: nearest untaken to bin center.
        if (best < 0) {
            const int64_t center = (bin_lo + bin_hi) / 2;
            int64_t best_dist = INT64_MAX;
            for (std::size_t i = 0; i < cands.size(); ++i) {
                if (taken[i]) continue;
                const int64_t mid = (cands[i].start_ms + cands[i].end_ms) / 2;
                const int64_t d = std::llabs(mid - center);
                if (d < best_dist ||
                    (d == best_dist && best >= 0 &&
                     cands[i].score >
                         cands[static_cast<std::size_t>(best)].score)) {
                    best_dist = d;
                    best = static_cast<int>(i);
                }
            }
        }
        if (best >= 0) {
            taken[static_cast<std::size_t>(best)] = true;
            picked.push_back(cands[static_cast<std::size_t>(best)]);
        }
    }

    // Fill remaining by score.
    std::vector<std::size_t> order(cands.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
        return cands[a].score > cands[b].score;
    });
    for (std::size_t idx : order) {
        if (static_cast<int>(picked.size()) >= target_n) break;
        if (taken[idx]) continue;
        // Skip overlaps with already picked.
        bool overlaps = false;
        for (const auto& p : picked) {
            if (!(cands[idx].end_ms <= p.start_ms ||
                  cands[idx].start_ms >= p.end_ms)) {
                overlaps = true;
                break;
            }
        }
        if (overlaps) continue;
        taken[idx] = true;
        picked.push_back(cands[idx]);
    }

    std::sort(picked.begin(), picked.end(),
              [](const Candidate& a, const Candidate& b) {
                  return a.start_ms < b.start_ms;
              });
    return picked;
}

[[nodiscard]] Segment to_segment(const std::vector<Word>& words,
                                 const Candidate& c, int index) {
    Segment s;
    s.index = index;
    s.start_ms = c.start_ms;
    s.end_ms = c.end_ms;
    s.words.assign(words.begin() + static_cast<std::ptrdiff_t>(c.span.begin),
                   words.begin() + static_cast<std::ptrdiff_t>(c.span.end));
    return s;
}

}  // namespace

std::vector<Segment> slice_segments(const std::vector<Word>& words,
                                    const SliceOptions& opt) {
    std::vector<Segment> result;
    if (words.empty()) return result;

    const int64_t target_ms = static_cast<int64_t>(opt.target_sec) * 1000;
    const int want = resolve_target_clips(words, opt);

    if (opt.verbose) {
        std::cout << "[Segmenter] target_clips=" << want
                  << " (requested=" << opt.target_clips << ") pause_ms="
                  << opt.pause_ms << " target_sec=" << opt.target_sec << '\n';
    }

    // --- Phase 1: speech regions ------------------------------------------
    const auto regions = build_speech_regions(words, opt.pause_ms);
    if (opt.verbose) {
        std::cout << "[Segmenter] speech regions: " << regions.size() << '\n';
        for (const auto& r : regions) {
            std::cout << "  region " << fmt_range(span_start(words, r),
                                                  span_end(words, r))
                      << "  words=" << (r.end - r.begin) << '\n';
        }
    }

    // --- Phase 2: candidates from each region -----------------------------
    std::vector<Candidate> candidates;
    for (const auto& region : regions) {
        auto pieces = cut_region(words, region, opt);
        for (const auto& piece : pieces) {
            Candidate c = make_candidate(words, piece, target_ms);
            if (!filter_candidate(c, opt, opt.verbose)) continue;
            candidates.push_back(c);
        }
    }

    if (opt.verbose) {
        std::cout << "[Segmenter] raw candidates after filter: "
                  << candidates.size() << '\n';
    }

    // --- Phase 3: hit target_clips ----------------------------------------
    if (static_cast<int>(candidates.size()) < want) {
        // Further split longest candidates until we reach want or stall.
        int guard = 0;
        while (static_cast<int>(candidates.size()) < want && guard++ < 64) {
            auto longest = std::max_element(
                candidates.begin(), candidates.end(),
                [](const Candidate& a, const Candidate& b) {
                    return (a.end_ms - a.start_ms) < (b.end_ms - b.start_ms);
                });
            if (longest == candidates.end()) break;
            if ((longest->end_ms - longest->start_ms) <
                static_cast<int64_t>(opt.min_sec) * 2000) {
                break;  // cannot usefully split further
            }

            std::vector<Candidate> parts;
            if (!try_split_candidate(words, *longest, parts, opt)) break;

            if (opt.verbose) {
                std::cout << "[Segmenter] split "
                          << fmt_range(longest->start_ms, longest->end_ms)
                          << " → " << parts.size() << " pieces\n";
            }
            candidates.erase(longest);
            for (auto& p : parts) candidates.push_back(std::move(p));
        }
    }

    std::vector<Candidate> chosen;
    if (static_cast<int>(candidates.size()) > want) {
        chosen = select_diverse(std::move(candidates), want);
    } else {
        chosen = std::move(candidates);
        std::sort(chosen.begin(), chosen.end(),
                  [](const Candidate& a, const Candidate& b) {
                      return a.start_ms < b.start_ms;
                  });
    }

    result.reserve(chosen.size());
    for (std::size_t i = 0; i < chosen.size(); ++i) {
        log_accepted(opt.verbose, chosen[i].start_ms, chosen[i].end_ms,
                     chosen[i].score);
        result.push_back(to_segment(words, chosen[i], static_cast<int>(i)));
    }

    return result;
}

}  // namespace autoclip
