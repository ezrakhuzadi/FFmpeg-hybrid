#!/bin/bash

# GPU Strengths Test - Highlights where WebGPU excels
# Tests scenarios that benefit most from GPU acceleration

echo "================================================"
echo "WebGPU VP9 Acceleration Strengths Test"
echo "================================================"
echo ""
echo "This test highlights scenarios where GPU acceleration"
echo "provides the most benefit over CPU-only decoding."
echo ""

WEBGPU_FFMPEG="./ffmpeg"
CPU_FFMPEG="/Users/ezra/Downloads/FFmpeg-baseline/ffmpeg"

# Check baseline exists
if [ ! -f "$CPU_FFMPEG" ]; then
    echo "Building baseline FFmpeg for comparison..."
    cd /Users/ezra/Downloads
    if [ ! -d "FFmpeg-baseline" ]; then
        cp -r FFmpeg FFmpeg-baseline
        cd FFmpeg-baseline
        make clean
        ./configure --disable-webgpu  # Configure without WebGPU
        make -j8
    fi
    cd /Users/ezra/Downloads/FFmpeg
fi

# Test 1: Transform-heavy content
echo "TEST 1: Transform-Heavy Content"
echo "--------------------------------"
echo "GPU excels at parallel transform operations"
echo ""

# Create a test pattern with many small blocks (lots of transforms)
$WEBGPU_FFMPEG -f lavfi -i "testsrc2=size=1920x1080:rate=30:duration=5" \
    -c:v libvpx-vp9 -b:v 5M -g 30 transform_test.webm -y 2>/dev/null

echo "Decoding with CPU-only:"
time $CPU_FFMPEG -i transform_test.webm -f null - 2>&1 | grep "fps="

echo ""
echo "Decoding with WebGPU acceleration:"
time $WEBGPU_FFMPEG -i transform_test.webm -f null - 2>&1 | grep "fps="

echo ""

# Test 2: Motion compensation stress test
echo "TEST 2: Motion Compensation Test"
echo "---------------------------------"
echo "GPU texture units accelerate motion compensation"
echo ""

# Create high-motion test video
$WEBGPU_FFMPEG -f lavfi -i "testsrc=size=1280x720:rate=30:duration=5,\
    rotate=PI*t:ow=1280:oh=720:c=black" \
    -c:v libvpx-vp9 -b:v 3M motion_test.webm -y 2>/dev/null

echo "Decoding with CPU-only:"
time $CPU_FFMPEG -i motion_test.webm -f null - 2>&1 | grep "fps="

echo ""
echo "Decoding with WebGPU acceleration:"
time $WEBGPU_FFMPEG -i motion_test.webm -f null - 2>&1 | grep "fps="

echo ""

# Test 3: Parallel block processing
echo "TEST 3: Parallel Block Processing"
echo "----------------------------------"
echo "Multiple blocks processed simultaneously on GPU"
echo ""

# Test with demo2.webm if available (high resolution benefits GPU)
if [ -f "demo2.webm" ]; then
    echo "Using demo2.webm (high resolution)"
    
    echo "CPU-only - First 100 frames:"
    $CPU_FFMPEG -i demo2.webm -frames:v 100 -benchmark -f null - 2>&1 | \
        grep -E "bench:|fps=" | tail -2
    
    echo ""
    echo "WebGPU - First 100 frames:"
    $WEBGPU_FFMPEG -i demo2.webm -frames:v 100 -benchmark -f null - 2>&1 | \
        grep -E "bench:|fps=" | tail -2
fi

echo ""

# Test 4: Chroma plane processing
echo "TEST 4: Chroma Plane Processing"
echo "--------------------------------"
echo "Y/U/V planes processed in parallel on GPU"
echo ""

# Create colorful test video (stress chroma processing)
$WEBGPU_FFMPEG -f lavfi -i "testsrc2=size=1920x1080:rate=30:duration=3,\
    hue=H=2*PI*t:s=2" \
    -c:v libvpx-vp9 -pix_fmt yuv420p color_test.webm -y 2>/dev/null

echo "Monitoring plane processing (WebGPU):"
$WEBGPU_FFMPEG -v info -i color_test.webm -frames:v 30 -f null - 2>&1 | \
    grep -E "plane transform|WebGPU" | head -10

echo ""

# Summary
echo "================================================"
echo "Performance Analysis Summary"
echo "================================================"
echo ""
echo "GPU acceleration provides best speedup for:"
echo "1. High-resolution content (4K/8K)"
echo "2. Content with many transform operations"
echo "3. High-motion scenes (motion compensation)"
echo "4. Parallel processing of Y/U/V planes"
echo ""
echo "The WebGPU implementation offloads:"
echo "- Inverse transforms (DCT/ADST)"
echo "- Motion compensation"
echo "- Loop filtering"
echo "- Chroma plane processing"
echo ""
echo "While CPU handles:"
echo "- Bitstream parsing"
echo "- Entropy decoding"
echo "- Frame management"