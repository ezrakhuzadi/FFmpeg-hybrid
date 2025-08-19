#!/bin/bash

echo "VP9 WebGPU Hardware Acceleration Test"
echo "======================================"
echo ""

# Test file
VIDEO="/Users/ezra/Downloads/demo.webm"

echo "1. Testing WITH WebGPU acceleration:"
echo "-------------------------------------"
time ./ffmpeg -i "$VIDEO" -t 5 -f null - 2>&1 | grep -E "WebGPU|fps"

echo ""
echo "2. Testing WITHOUT hardware acceleration (reference):"
echo "------------------------------------------------------"
echo "(Would need to rebuild without WebGPU to compare)"

echo ""
echo "3. Detailed GPU operation log (first 10 seconds):"
echo "---------------------------------------------------"
./ffmpeg -v info -i "$VIDEO" -t 10 -f null - 2>&1 | grep -E "WebGPU|GPU" | head -20

echo ""
echo "Test complete!"