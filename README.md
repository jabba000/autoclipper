# 🎬 AutoClipper

![C++](https://img.shields.io/badge/Language-C++-00599C?style=for-the-badge&logo=c%2B%2B)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**An automated, local-first vertical video clipper for YouTube Shorts, TikTok, and Reels.**



## ✨ Key Features

* **Intelligent AI Slicing:** Leverages Whisper.cpp for high-accuracy transcription and semantically chunks videos at natural pauses.


* **Dynamic Face Tracking:** Utilizes OpenCV DNN (`res10_300x300_ssd`) to ensure the dominant speaker remains perfectly centered in a vertical 9:16 frame.


* **TikTok-Style Subtitles:** Automatically generates and burns in lower-third, word-by-word karaoke ASS subtitles featuring Arial Black with a high-visibility yellow and white color scheme.


* **Hardware-Accelerated Rendering:** Speeds up FFmpeg video encoding with native, out-of-process support for NVENC, VAAPI, QSV, and x264 fallbacks.


* **Concurrent Processing:** Implements a bounded thread pool to process facial tracking, subtitle generation, and rendering jobs in parallel.



## 🛠️ Prerequisites

To build and run AutoClipper, your system requires the following components:

* **Core Build Tools:** CMake (3.20+) and a C++20 compatible compiler.


* **System Libraries:** FFmpeg (accessible via system PATH) and OpenCV (specifically the `core`, `imgproc`, `videoio`, and `dnn` components).


* **AI Models:**
* Whisper GGML model (e.g., `ggml-medium.bin`).


* OpenCV Face DNN prototxt (`deploy.prototxt`).


* OpenCV Face DNN weights (`res10_300x300_ssd_iter_140000_fp16.caffemodel`).





## 🚀 Build Instructions

AutoClipper is built using CMake. Run **b.sh** (./b.sh) or execute the following terminal commands inside the AutoClipper folder to compile the project cleanly:

```bash
# Clean previous builds
rm -rf build
mkdir -p build

# Configure the project in Release mode
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Compile utilizing all available CPU cores
cmake --build build -j"$(nproc)"

```

## 💻 Usage Guide

The compiled binary will be located in the `build` directory. You can use **do.sh** (./d.sh) or the standard CLI command structure to process a video:

```bash
./build/autoclipper -i video.mp4 \
    -m extra/ggml-medium.bin \
    -o ./output \
    -e nvenc \
    --face-proto extra/deploy.prototxt \
    --face-model extra/res10_300x300_ssd_iter_140000_fp16.caffemodel \
    --target-clips 8

```

### 🎛️ Core CLI Arguments

* `-i`, `--input`: **(Required)** Path to the input video file.


* `-o`, `--output`: Output directory for the sliced clips (defaults to `./output`).


* `-m`, `--model`: Path to the Whisper GGML model (defaults to `extra/ggml-base.bin`).


* `-e`, `--encoder`: Select the hardware video encoder: `x264`, `nvenc`, `vaapi`, or `qsv` (defaults to `x264`).


* `--face-proto` / `--face-model`: Paths to the OpenCV Caffe SSD face tracking files. *Note: Without these files, the software falls back to a static centered 9:16 crop.*


* `--target-clips`: Target number of short clips to extract (set to `0` for automatic calculation based on timeline duration).

• *x264 (CPU processing, slow)*
• *nvenc (hardware acceleration using NVIDIA graphics cards)*
• *vaapi (hardware acceleration using AMD/Intel)*


## 🗺️ Roadmap

* [x] **Ukrainian Language Support:** Fully functional transcription with localized punctuation mapping, silence filtering, and text merging.
* [ ] **English Localization:** Trivial integration planned for the immediate roadmap to expand transcription and word-boundary rules.



## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

### Summary of Terms:
* **Attribution:** You must retain original copyright notices and give appropriate credit.
* **Patent Protection:** Includes an explicit grant of patent rights from contributors and protects against patent litigation.
* **State Changes:** Any modified files must carry prominent notices stating that they were altered.
