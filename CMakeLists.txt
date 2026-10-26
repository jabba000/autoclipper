cmake_minimum_required(VERSION 3.20)
project(AutoClipper
    VERSION 1.0.0
    DESCRIPTION "Local OpusClip-style vertical video clipper"
    LANGUAGES CXX C
)

# License: Apache License 2.0

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Aggressive release flags for the clipper itself (not forced onto deps).
add_compile_options(
    $<$<CONFIG:Release>:-O3>
    $<$<CONFIG:Release>:-DNDEBUG>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wextra>
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wpedantic>
)

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio dnn)
find_package(Threads REQUIRED)

# whisper.cpp via FetchContent — disable demos/tests for a lean build.
include(FetchContent)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    whisper
    GIT_REPOSITORY https://github.com/ggerganov/whisper.cpp.git
    GIT_TAG        v1.7.5
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(whisper)

# ---------------------------------------------------------------------------
# AutoClipper
# ---------------------------------------------------------------------------
add_executable(autoclipper
    src/main.cpp
    src/cli.cpp
    src/process.cpp
    src/whisper_transcriber.cpp
    src/segmenter.cpp
    src/face_tracker.cpp
    src/ass_generator.cpp
    src/renderer.cpp
)

target_include_directories(autoclipper PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(autoclipper PRIVATE
    whisper
    ${OpenCV_LIBS}
    Threads::Threads
)

# posix_spawn lives in libc on Linux; nothing extra to link.

install(TARGETS autoclipper RUNTIME DESTINATION bin)

message(STATUS "OpenCV: ${OpenCV_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

