#pragma once

/**
 * OpenCV DNN face tracker (res10_300x300 SSD)
 * -------------------------------------------
 * Supply the Caffe pair shipped under extra/:
 *   - deploy.prototxt
 *   - res10_300x300_ssd_iter_140000_fp16.caffemodel
 *
 * Or download from OpenCV's face detector model zoo:
 *   https://github.com/opencv/opencv/tree/master/samples/dnn/face_detector
 *
 * Usage (CLI):
 *   --face-proto extra/deploy.prototxt
 *   --face-model extra/res10_300x300_ssd_iter_140000_fp16.caffemodel
 *
 * Sampling: 1 frame per second across the segment; face centers are averaged
 * (EMA-smoothed) then expanded to a 9:16 crop. Falls back to center-crop
 * when confidence stays below the threshold.
 *
 * Thread safety: FaceTracker is NOT thread-safe. Create one instance per
 * worker, or serialize detect() calls behind a mutex.
 */

#include "types.hpp"

#include <filesystem>
#include <mutex>
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <string>

namespace autoclip {

namespace fs = std::filesystem;

class FaceTracker {
public:
    FaceTracker() = default;

    /**
     * Load MobileNet-SSD / res10 Caffe model. Throws on failure.
     * Empty paths leave the tracker in "center-crop only" mode.
     */
    void load(const fs::path& proto, const fs::path& model);

    [[nodiscard]] bool loaded() const noexcept { return loaded_; }

    /**
     * Sample frames across [start_ms, end_ms] and return a smoothed 9:16 crop.
     * Target output framing is 1080x1920 (aspect only — absolute size from source).
     */
    [[nodiscard]] CropRect compute_crop(
        const fs::path& video_path,
        int64_t start_ms,
        int64_t end_ms);

private:
    [[nodiscard]] CropRect center_crop(int vid_w, int vid_h) const;
    [[nodiscard]] float detect_face_center_x(const cv::Mat& frame, int vid_w);

    cv::dnn::Net net_;
    bool loaded_ = false;
    std::mutex mutex_;  // OpenCV DNN forward is not guaranteed re-entrant
    static constexpr float kConfThresh = 0.5f;
    static constexpr float kPadRatio = 0.08f;  // slight horizontal padding bias
};

}  // namespace autoclip
