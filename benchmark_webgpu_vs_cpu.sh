#!/bin/bash

# WebGPU VP9 vs CPU-only Decoder Benchmark
# This test compares the WebGPU-accelerated FFmpeg against baseline CPU-only FFmpeg

echo "=========================================="
echo "WebGPU VP9 vs CPU-only Decoder Benchmark"
echo "=========================================="
echo ""

# Paths
WEBGPU_FFMPEG="./ffmpeg"
CPU_FFMPEG="/Users/ezra/Downloads/FFmpeg-baseline/ffmpeg"

# Test files (create if needed)
TEST_FILE="demo2.webm"

# Check if both FFmpeg builds exist
if [ ! -f "$WEBGPU_FFMPEG" ]; then
    echo "Error: WebGPU FFmpeg not found at $WEBGPU_FFMPEG"
    exit 1
fi

if [ ! -f "$CPU_FFMPEG" ]; then
    echo "Error: CPU-only FFmpeg not found at $CPU_FFMPEG"
    echo "Please ensure FFmpeg-baseline is built in /Users/ezra/Downloads/FFmpeg-baseline/"
    exit 1
fi

if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file $TEST_FILE not found"
    echo "Attempting to download a test file..."
    # Download Big Buck Bunny in VP9 if no test file exists
    curl -L "https://test-videos.co.uk/vids/bigbuckbunny/webm/vp9/360/Big_Buck_Bunny_360_10s_1MB.webm" -o test_360p.webm
    TEST_FILE="test_360p.webm"
fi

echo "Test Configuration:"
echo "- WebGPU FFmpeg: $WEBGPU_FFMPEG"
echo "- CPU FFmpeg: $CPU_FFMPEG"
echo "- Test File: $TEST_FILE"
echo ""

# Function to run benchmark
run_benchmark() {
    local ffmpeg_binary=$1
    local test_name=$2
    local test_file=$3
    local duration=$4
    
    echo "Running $test_name..."
    
    # Warm-up run
    $ffmpeg_binary -i "$test_file" -t 1 -f null - 2>/dev/null
    
    # Actual benchmark (3 runs)
    local total_time=0
    local fps_sum=0
    
    for i in 1 2 3; do
        echo -n "  Run $i: "
        
        # Run FFmpeg and capture output
        local output=$( { time $ffmpeg_binary -i "$test_file" -t "$duration" -benchmark -f null - 2>&1; } 2>&1 )
        
        # Extract FPS from output (look for "frame=" line)
        local fps=$(echo "$output" | grep -oE 'fps=[0-9.]+' | tail -1 | cut -d'=' -f2)
        
        # Extract real time from output
        local real_time=$(echo "$output" | grep "^real" | awk '{print $2}')
        
        # Extract benchmark time if available
        local bench_time=$(echo "$output" | grep "bench:" | grep -oE 'utime=[0-9.]+s' | cut -d'=' -f2 | sed 's/s//')
        
        if [ -z "$fps" ]; then
            fps="0"
        fi
        
        echo "FPS: $fps, Time: $real_time"
        
        # Add to sum for averaging
        fps_sum=$(echo "$fps_sum + $fps" | bc -l)
    done
    
    local avg_fps=$(echo "scale=2; $fps_sum / 3" | bc -l)
    echo "  Average FPS: $avg_fps"
    echo ""
    
    return 0
}

# Test 1: Basic decode performance
echo "TEST 1: Basic Decode Performance (10 seconds)"
echo "----------------------------------------------"

run_benchmark "$CPU_FFMPEG" "CPU-only decoder" "$TEST_FILE" 10
CPU_FPS=$avg_fps

run_benchmark "$WEBGPU_FFMPEG" "WebGPU-accelerated decoder" "$TEST_FILE" 10
WEBGPU_FPS=$avg_fps

# Calculate speedup
if [ $(echo "$CPU_FPS > 0" | bc -l) -eq 1 ]; then
    SPEEDUP=$(echo "scale=2; $WEBGPU_FPS / $CPU_FPS" | bc -l)
    echo "Speedup: ${SPEEDUP}x"
fi

echo ""

# Test 2: Memory bandwidth test (repeated decode)
echo "TEST 2: Memory Bandwidth Test (5 passes)"
echo "-----------------------------------------"

echo "CPU-only decoder:"
time for i in {1..5}; do
    $CPU_FFMPEG -i "$TEST_FILE" -t 5 -f null - 2>/dev/null
done

echo ""
echo "WebGPU-accelerated decoder:"
time for i in {1..5}; do
    $WEBGPU_FFMPEG -i "$TEST_FILE" -t 5 -f null - 2>/dev/null
done

echo ""

# Test 3: CPU usage comparison
echo "TEST 3: CPU Usage Comparison"
echo "-----------------------------"

# Function to measure CPU usage
measure_cpu() {
    local ffmpeg_binary=$1
    local test_name=$2
    
    echo "$test_name CPU usage:"
    
    # Run in background and get PID
    $ffmpeg_binary -i "$TEST_FILE" -t 10 -f null - 2>/dev/null &
    local pid=$!
    
    # Monitor CPU usage
    local max_cpu=0
    local samples=0
    
    while kill -0 $pid 2>/dev/null; do
        local cpu=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ')
        if [ ! -z "$cpu" ]; then
            samples=$((samples + 1))
            echo -n "."
            if (( $(echo "$cpu > $max_cpu" | bc -l) )); then
                max_cpu=$cpu
            fi
        fi
        sleep 0.1
    done
    
    echo ""
    echo "  Peak CPU usage: ${max_cpu}%"
    echo ""
}

measure_cpu "$CPU_FFMPEG" "CPU-only"
measure_cpu "$WEBGPU_FFMPEG" "WebGPU"

# Test 4: Profile detection (WebGPU only)
echo "TEST 4: VP9 Profile Detection (WebGPU)"
echo "---------------------------------------"
$WEBGPU_FFMPEG -i "$TEST_FILE" -t 1 -f null - 2>&1 | grep -E "Profile|bit depth|WebGPU" | head -10

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Key metrics to compare:"
echo "1. FPS (higher is better) - Shows decode speed"
echo "2. CPU usage (lower is better) - Shows efficiency"
echo "3. Speedup factor - Direct performance comparison"
echo ""
echo "Expected results:"
echo "- WebGPU should show higher FPS on high-resolution content"
echo "- WebGPU should use less CPU (offloading to GPU)"
echo "- Speedup should be >1.0x, especially for 4K+ content"