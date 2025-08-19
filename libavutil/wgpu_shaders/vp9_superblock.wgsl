// VP9 Superblock Processing WebGPU Compute Shader
// Processes entire 64x64 superblocks with all their transforms in a single dispatch
// Optimized for GPU parallelism

struct SuperblockInfo {
    // Position of superblock in frame
    sb_x: u32,
    sb_y: u32,
    
    // Partition info (how the 64x64 is split)
    partition_mask: u32,  // Bit mask indicating partition structure
    
    // Number of transforms in this superblock
    num_transforms: u32,
    
    // Offset into transform list
    transform_offset: u32,
    
    // Padding
    _pad: array<u32, 3>,
}

struct TransformInfo {
    // Position within superblock (0-63)
    local_x: u32,
    local_y: u32,
    
    // Transform size (0=4x4, 1=8x8, 2=16x16, 3=32x32)
    tx_size: u32,
    
    // Transform type
    tx_type: u32,
    
    // Coefficient offset
    coeff_offset: u32,
    
    // End of block position
    eob: u32,
    
    // Padding
    _pad: array<u32, 2>,
}

// Bindings
@group(0) @binding(0) var<storage, read> superblocks: array<SuperblockInfo>;
@group(0) @binding(1) var<storage, read> transforms: array<TransformInfo>;
@group(0) @binding(2) var<storage, read> coefficients: array<i32>;
@group(0) @binding(3) var<storage, read_write> frame_buffer: array<u32>;  // Frame pixels
@group(0) @binding(4) var<storage, read> prediction: array<u32>;  // Prediction pixels
@group(0) @binding(5) var<uniform> frame_info: FrameInfo;

struct FrameInfo {
    width: u32,
    height: u32,
    stride_y: u32,
    stride_uv: u32,
    bit_depth: u32,
    _pad: array<u32, 3>,
}

// Shared memory for superblock processing
var<workgroup> sb_pixels: array<array<i32, 64>, 64>;  // 64x64 superblock pixels
var<workgroup> sb_coeffs: array<i32, 4096>;  // Max coefficients for superblock

// Fast 4x4 IDCT
fn idct4x4(coeffs: ptr<function, array<i32, 16>>) -> array<i32, 16> {
    var output: array<i32, 16>;
    var temp: array<i32, 16>;
    
    // Row transform
    for (var i = 0u; i < 4u; i++) {
        let c0 = (*coeffs)[i * 4u + 0u];
        let c1 = (*coeffs)[i * 4u + 1u];
        let c2 = (*coeffs)[i * 4u + 2u];
        let c3 = (*coeffs)[i * 4u + 3u];
        
        let t0 = (c0 + c2) * 64;
        let t1 = (c0 - c2) * 64;
        let t2 = c1 * 83 + c3 * 36;
        let t3 = c1 * 36 - c3 * 83;
        
        temp[i * 4u + 0u] = (t0 + t2) >> 7;
        temp[i * 4u + 1u] = (t1 + t3) >> 7;
        temp[i * 4u + 2u] = (t1 - t3) >> 7;
        temp[i * 4u + 3u] = (t0 - t2) >> 7;
    }
    
    // Column transform
    for (var i = 0u; i < 4u; i++) {
        let c0 = temp[0u * 4u + i];
        let c1 = temp[1u * 4u + i];
        let c2 = temp[2u * 4u + i];
        let c3 = temp[3u * 4u + i];
        
        let t0 = (c0 + c2) * 64;
        let t1 = (c0 - c2) * 64;
        let t2 = c1 * 83 + c3 * 36;
        let t3 = c1 * 36 - c3 * 83;
        
        output[0u * 4u + i] = (t0 + t2 + 2048) >> 12;
        output[1u * 4u + i] = (t1 + t3 + 2048) >> 12;
        output[2u * 4u + i] = (t1 - t3 + 2048) >> 12;
        output[3u * 4u + i] = (t0 - t2 + 2048) >> 12;
    }
    
    return output;
}

// Process a single transform within the superblock
fn process_transform(transform_idx: u32) {
    let transform = transforms[transform_idx];
    let tx_size = transform.tx_size;
    let local_x = transform.local_x;
    let local_y = transform.local_y;
    let coeff_offset = transform.coeff_offset;
    let eob = transform.eob;
    
    if (eob == 0u) {
        return;  // No coefficients to process
    }
    
    // Process based on transform size
    if (tx_size == 0u) {  // 4x4
        var coeffs: array<i32, 16>;
        
        // Load coefficients
        for (var i = 0u; i < min(16u, eob); i++) {
            coeffs[i] = coefficients[coeff_offset + i];
        }
        
        // Apply IDCT
        let residuals = idct4x4(&coeffs);
        
        // Add to superblock pixels (with prediction)
        for (var y = 0u; y < 4u; y++) {
            for (var x = 0u; x < 4u; x++) {
                let px = local_x + x;
                let py = local_y + y;
                let idx = y * 4u + x;
                
                // Add residual to prediction
                sb_pixels[py][px] += residuals[idx];
            }
        }
    }
    // TODO: Add 8x8, 16x16, 32x32 transforms
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let sb_idx = workgroup_id.x + workgroup_id.y * (frame_info.width / 64u);
    
    if (sb_idx >= arrayLength(&superblocks)) {
        return;
    }
    
    let sb = superblocks[sb_idx];
    let sb_x = sb.sb_x;
    let sb_y = sb.sb_y;
    
    // Initialize superblock pixels with prediction
    let local_x = local_id.x * 8u;
    let local_y = local_id.y * 8u;
    
    // Each thread loads 8x8 pixels from prediction
    for (var dy = 0u; dy < 8u; dy++) {
        for (var dx = 0u; dx < 8u; dx++) {
            let px = sb_x * 64u + local_x + dx;
            let py = sb_y * 64u + local_y + dy;
            
            if (px < frame_info.width && py < frame_info.height) {
                let pred_idx = py * frame_info.stride_y + px;
                sb_pixels[local_y + dy][local_x + dx] = i32(prediction[pred_idx / 4u]);
            }
        }
    }
    
    workgroupBarrier();
    
    // Process all transforms in this superblock
    // Each thread processes a subset of transforms
    let transforms_per_thread = (sb.num_transforms + 63u) / 64u;
    let thread_idx = local_id.y * 8u + local_id.x;
    
    for (var i = 0u; i < transforms_per_thread; i++) {
        let transform_idx = sb.transform_offset + thread_idx + i * 64u;
        if (transform_idx < sb.transform_offset + sb.num_transforms) {
            process_transform(transform_idx);
        }
    }
    
    workgroupBarrier();
    
    // Write back to frame buffer
    for (var dy = 0u; dy < 8u; dy++) {
        for (var dx = 0u; dx < 8u; dx++) {
            let px = sb_x * 64u + local_x + dx;
            let py = sb_y * 64u + local_y + dy;
            
            if (px < frame_info.width && py < frame_info.height) {
                let frame_idx = py * frame_info.stride_y + px;
                let pixel = clamp(sb_pixels[local_y + dy][local_x + dx], 0, 255);
                
                // Pack into u32 (assuming 8-bit for now)
                let shift = (frame_idx % 4u) * 8u;
                let word_idx = frame_idx / 4u;
                atomicOr(&frame_buffer[word_idx], u32(pixel) << shift);
            }
        }
    }
}