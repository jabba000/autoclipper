#!/bin/bash

./build/autoclipper -i video.mp4 -m extra/ggml-medium.bin -o ./output \
    -e x264|nvenc|vaapi|qsv \
    --face-proto extra/deploy.prototxt \
    --face-model extra/res10_300x300_ssd_iter_140000_fp16.caffemodel \
    --target-clips 8
