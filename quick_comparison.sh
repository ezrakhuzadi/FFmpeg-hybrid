#!/bin/bash

# Quick A/B Comparison: WebGPU vs CPU-only VP9 Decoder

echo "Quick VP9 Decoder Comparison"
echo "============================="
echo ""

# Build baseline if needed
if [ ! -f "/Users/ezra/Downloads/FFmpeg-baseline/ffmpeg" ]; then
    echo "Setting up baseline FFmpeg (CPU-only)..."
    cd /Users/ezra/Downloads
    cp -r FFmpeg FFmpeg-baseline
    cd FFmpeg-baseline
    git checkout HEAD -- .  # Reset any changes
    make clean
    ./configure  # Default config without WebGPU
    make -j8
    cd /Users/ezra/Downloads/FFmpeg
fi

# Simple speed test
echo "Speed Test - Decoding demo2.webm (first 5 seconds):"
echo ""

if [ -f "demo2.webm" ]; then
    echo "1. CPU-only decoder:"
    echo "--------------------"
    /usr/bin/time -l /Users/ezra/Downloads/FFmpeg-baseline/ffmpeg \
        -i demo2.webm -t 5 -f null - 2>&1 | \
        grep -E "real|user|sys|fps=|maximum resident" | tail -5
    
    echo ""
    echo "2. WebGPU-accelerated decoder:"
    echo "------------------------------"
    /usr/bin/time -l ./ffmpeg \
        -i demo2.webm -t 5 -f null - 2>&1 | \
        grep -E "real|user|sys|fps=|maximum resident|WebGPU|Profile" | tail -8
else
    echo "Creating test file..."
    ./ffmpeg -f lavfi -i testsrc2=s=1920x1080:d=10:r=30 \
        -c:v libvpx-vp9 -b:v 5M test.webm -y
    
    echo "1. CPU-only decoder:"
    /usr/bin/time /Users/ezra/Downloads/FFmpeg-baseline/ffmpeg \
        -i test.webm -f null - 2>&1 | \
        grep -E "real|user|sys|fps=" | tail -4
    
    echo ""
    echo "2. WebGPU-accelerated decoder:"
    /usr/bin/time ./ffmpeg \
        -i test.webm -f null - 2>&1 | \
        grep -E "real|user|sys|fps=|WebGPU" | tail -5
fi

echo ""
echo "Key Metrics:"
echo "- 'real' time: Total wall clock time (lower is better)"
echo "- 'user' time: CPU time in user mode (lower means less CPU usage)"
echo "- 'fps': Frames per second processed (higher is better)"
echo "- Memory usage: Maximum resident set size"
echo ""
echo "Expected advantages of WebGPU:"
echo "✓ Higher FPS (faster processing)"
echo "✓ Lower CPU usage (work offloaded to GPU)"
echo "✓ Better scaling with resolution"