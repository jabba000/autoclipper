#include "face_tracker.hpp"

#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace autoclip {

void FaceTracker::load(const fs::path& proto, const fs::path& model) {
    if (proto.empty() || model.empty()) {
        loaded_ = false;
        return;
    }
    if (!fs::exists(proto) || !fs::exists(model)) {
        std::cerr << "[Warn] Face DNN files missing — using center crop.\n";
        loaded_ = false;
        return;
    }
    try {
        net_ = cv::dnn::readNetFromCaffe(proto.string(), model.string());
        // Prefer OpenCL / CPU backend — portable across machines without CUDA.
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        loaded_ = true;
    } catch (const cv::Exception& e) {
        std::cerr << "[Warn] Failed to load face DNN: " << e.what()
                  << " — using center crop.\n";
        loaded_ = false;
    }
}

CropRect FaceTracker::center_crop(int vid_w, int vid_h) const {
    // Full-height 9:16 window centered horizontally.
    int crop_h = vid_h;
    int crop_w = (vid_h * 9) / 16;
    if (crop_w > vid_w) {
        // Source is already narrower than 9:16 — letterbox by height.
        crop_w = vid_w;
        crop_h = (vid_w * 16) / 9;
    }
    int x = std::max(0, (vid_w - crop_w) / 2);
    int y = std::max(0, (vid_h - crop_h) / 2);
    crop_w = std::min(crop_w, vid_w - x);
    crop_h = std::min(crop_h, vid_h - y);
    return CropRect{x, y, crop_w, crop_h};
}

float FaceTracker::detect_face_center_x(const cv::Mat& frame, int vid_w) {
    if (!loaded_ || frame.empty()) return -1.f;

    std::lock_guard lock(mutex_);
    cv::Mat blob = cv::dnn::blobFromImage(
        frame, 1.0, cv::Size(300, 300), cv::Scalar(104.0, 177.0, 123.0), false,
        false);
    net_.setInput(blob);
    cv::Mat det = net_.forward();

    // Output shape: [1, 1, N, 7]
    cv::Mat detMat(det.size[2], det.size[3], CV_32F, det.ptr<float>());

    float best_conf = 0.f;
    float best_cx = -1.f;
    float best_area = 0.f;

    for (int i = 0; i < detMat.rows; ++i) {
        const float conf = detMat.at<float>(i, 2);
        if (conf < kConfThresh || conf < best_conf) continue;

        const float x1 = detMat.at<float>(i, 3) * static_cast<float>(vid_w);
        const float x2 = detMat.at<float>(i, 5) * static_cast<float>(vid_w);
        const float y1 = detMat.at<float>(i, 4) * static_cast<float>(frame.rows);
        const float y2 = detMat.at<float>(i, 6) * static_cast<float>(frame.rows);
        const float area = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);

        // Prefer higher confidence; break ties by larger face (dominant speaker).
        if (conf > best_conf || (conf == best_conf && area > best_area)) {
            best_conf = conf;
            best_area = area;
            best_cx = (x1 + x2) * 0.5f;
            (void)y1;
            (void)y2;
        }
    }
    return best_cx;
}

CropRect FaceTracker::compute_crop(const fs::path& video_path, int64_t start_ms,
                                   int64_t end_ms) {
    cv::VideoCapture cap(video_path.string());
    if (!cap.isOpened()) {
        throw std::runtime_error("Cannot open video for face tracking: " +
                                 video_path.string());
    }

    const int vid_w = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    const int vid_h = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    if (vid_w <= 0 || vid_h <= 0) {
        throw std::runtime_error("Invalid video dimensions");
    }

    CropRect fallback = center_crop(vid_w, vid_h);
    if (!loaded_) return fallback;

    // Sample ~1 fps across the segment (cap at 30 samples for long clips).
    const int64_t dur = std::max<int64_t>(1000, end_ms - start_ms);
    const int n_samples = static_cast<int>(
        std::clamp(dur / 1000, int64_t{1}, int64_t{30}));

    std::vector<float> centers;
    centers.reserve(static_cast<std::size_t>(n_samples));

    for (int s = 0; s < n_samples; ++s) {
        const int64_t t =
            start_ms + (dur * s) / std::max(1, n_samples - (n_samples > 1 ? 1 : 0));
        cap.set(cv::CAP_PROP_POS_MSEC, static_cast<double>(t));
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) continue;

        const float cx = detect_face_center_x(frame, vid_w);
        if (cx >= 0.f) centers.push_back(cx);
    }

    if (centers.empty()) return fallback;

    // EMA smoothing to damp frame-to-frame jitter, then mean of smoothed series.
    constexpr float alpha = 0.35f;
    float ema = centers.front();
    double sum = ema;
    for (std::size_t i = 1; i < centers.size(); ++i) {
        ema = alpha * centers[i] + (1.f - alpha) * ema;
        sum += ema;
    }
    const float smooth_cx = static_cast<float>(sum / static_cast<double>(centers.size()));

    int crop_h = vid_h;
    int crop_w = (vid_h * 9) / 16;
    if (crop_w > vid_w) {
        crop_w = vid_w;
        crop_h = (vid_w * 16) / 9;
    }

    // Slight padding bias: shift crop so face isn't glued to an edge.
    const int pad = static_cast<int>(crop_w * kPadRatio);
    int crop_x = static_cast<int>(std::lround(smooth_cx)) - crop_w / 2;
    crop_x = std::clamp(crop_x, 0, std::max(0, vid_w - crop_w));
    // Soft edge padding: if face near border, ease away a bit.
    if (crop_x < pad) crop_x = 0;
    if (crop_x > vid_w - crop_w - pad) crop_x = vid_w - crop_w;

    int crop_y = std::max(0, (vid_h - crop_h) / 2);
    return CropRect{crop_x, crop_y, crop_w, crop_h};
}

}  // namespace autoclip
