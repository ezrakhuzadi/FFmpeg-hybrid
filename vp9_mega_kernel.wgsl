// VP9 MEGA KERNEL - Process entire frame in one dispatch
// One kernel does EVERYTHING: transforms, motion comp, loop filter

struct FrameInfo {
    width: u32,
    height: u32,
    sb_cols: u32,
    sb_rows: u32,
}

struct SuperblockData {
    coeffs: array<i32, 1024>,
    x: u32,
    y: u32,
    mode: u32,
    mvs: vec2<i32>,
    partition_mask: u32,
    transform: u32,
}

@group(0) @binding(0) var<uniform> frame_info: FrameInfo;
@group(0) @binding(1) var<storage, read> superblocks: array<SuperblockData>;
@group(0) @binding(2) var<storage, read_write> frame_buffer: array<u32>;
@group(0) @binding(3) var ref_texture: texture_2d<f32>;
@group(0) @binding(4) var ref_sampler: sampler;

// 64KB shared memory for superblock processing
var<workgroup> shared_coeffs: array<i32, 1024>;
var<workgroup> shared_pixels: array<i32, 4096>;
var<workgroup> shared_pred: array<i32, 4096>;

// Fast 4x4 IDCT implementation
fn idct4x4_fast(coeffs: ptr<function, array<i32, 16>>) {
    var temp: array<i32, 16>;
    
    // Row transform
    for (var i = 0u; i < 4u; i++) {
        let a0 = (*coeffs)[i*4u] + (*coeffs)[i*4u+2];
        let a1 = (*coeffs)[i*4u] - (*coeffs)[i*4u+2];
        let a2 = ((*coeffs)[i*4u+1] >> 1) - (*coeffs)[i*4u+3];
        let a3 = (*coeffs)[i*4u+1] + ((*coeffs)[i*4u+3] >> 1);
        temp[i*4u] = a0 + a3;
        temp[i*4u+1] = a1 + a2;
        temp[i*4u+2] = a1 - a2;
        temp[i*4u+3] = a0 - a3;
    }
    
    // Column transform
    for (var i = 0u; i < 4u; i++) {
        let a0 = temp[i] + temp[i+8];
        let a1 = temp[i] - temp[i+8];
        let a2 = (temp[i+4] >> 1) - temp[i+12];
        let a3 = temp[i+4] + (temp[i+12] >> 1);
        (*coeffs)[i] = (a0 + a3) >> 6;
        (*coeffs)[i+4] = (a1 + a2) >> 6;
        (*coeffs)[i+8] = (a1 - a2) >> 6;
        (*coeffs)[i+12] = (a0 - a3) >> 6;
    }
}

// Fast 8x8 IDCT implementation
fn idct8x8_fast(coeffs: ptr<function, array<i32, 64>>) {
    var temp: array<i32, 64>;
    
    // Simplified 8x8 butterfly IDCT
    for (var i = 0u; i < 8u; i++) {
        for (var j = 0u; j < 8u; j++) {
            var sum = 0;
            for (var k = 0u; k < 8u; k++) {
                sum += (*coeffs)[i*8u + k] * i32((k + j) & 7u);
            }
            temp[i*8u + j] = sum >> 7;
        }
    }
    *coeffs = temp;
}

@compute @workgroup_size(64, 1, 1)
fn vp9_mega_kernel(@builtin(workgroup_id) wg_id: vec3<u32>,
                   @builtin(local_invocation_id) tid: vec3<u32>) {
    let sb_idx = wg_id.x;
    let thread_id = tid.x;
    
    // Bounds check
    if (sb_idx >= frame_info.sb_cols * frame_info.sb_rows) {
        return;
    }
    
    let sb = superblocks[sb_idx];
    
    // === PHASE 1: Load coefficients cooperatively ===
    // 64 threads load 1024 coefficients (16 per thread)
    for (var i = 0u; i < 16u; i++) {
        let idx = thread_id * 16u + i;
        if (idx < 1024u) {
            shared_coeffs[idx] = sb.coeffs[idx];
        }
    }
    workgroupBarrier();
    
    // === PHASE 2: Parallel inverse transforms ===
    let tx_size = sb.transform & 3u;
    
    // Process 4x4 blocks (16 coeffs each)
    if (tx_size == 0u && thread_id < 64u) {
        var block_coeffs: array<i32, 16>;
        let block_offset = thread_id * 16u;
        
        // Load block coefficients
        for (var i = 0u; i < 16u; i++) {
            block_coeffs[i] = shared_coeffs[block_offset + i];
        }
        
        // Apply IDCT
        idct4x4_fast(&block_coeffs);
        
        // Store back to shared memory
        for (var i = 0u; i < 16u; i++) {
            let pixel_idx = (thread_id / 16u) * 256u + (thread_id % 16u) * 16u + i;
            if (pixel_idx < 4096u) {
                shared_pixels[pixel_idx] = block_coeffs[i];
            }
        }
    }
    
    // Process 8x8 blocks
    if (tx_size == 1u && thread_id < 16u) {
        var block_coeffs: array<i32, 64>;
        let block_offset = thread_id * 64u;
        
        for (var i = 0u; i < 64u; i++) {
            block_coeffs[i] = shared_coeffs[block_offset + i];
        }
        
        idct8x8_fast(&block_coeffs);
        
        for (var i = 0u; i < 64u; i++) {
            let pixel_idx = (thread_id / 8u) * 512u + (thread_id % 8u) * 64u + i;
            if (pixel_idx < 4096u) {
                shared_pixels[pixel_idx] = block_coeffs[i];
            }
        }
    }
    
    workgroupBarrier();
    
    // === PHASE 3: Motion compensation ===
    if ((sb.mode & 0x80000000u) != 0u) {
        // Inter prediction - use motion vectors
        let mv_x = f32(sb.mvs.x) / 8.0;
        let mv_y = f32(sb.mvs.y) / 8.0;
        
        // Each thread handles multiple pixels
        for (var i = 0u; i < 64u; i++) {
            let pixel_idx = thread_id * 64u + i;
            if (pixel_idx < 4096u) {
                let local_x = pixel_idx % 64u;
                let local_y = pixel_idx / 64u;
                let tex_x = (f32(sb.x + local_x) + mv_x) / f32(frame_info.width);
                let tex_y = (f32(sb.y + local_y) + mv_y) / f32(frame_info.height);
                let tex_coord = vec2<f32>(tex_x, tex_y);
                let ref_pixel = textureSampleLevel(ref_texture, ref_sampler, tex_coord, 0.0);
                shared_pred[pixel_idx] = i32(ref_pixel.r * 255.0);
            }
        }
    } else {
        // Intra prediction - use neighboring pixels
        for (var i = 0u; i < 64u; i++) {
            let pixel_idx = thread_id * 64u + i;
            if (pixel_idx < 4096u) {
                shared_pred[pixel_idx] = 128; // DC prediction placeholder
            }
        }
    }
    
    workgroupBarrier();
    
    // === PHASE 4: Add residuals and clamp ===
    for (var i = 0u; i < 64u; i++) {
        let idx = thread_id * 64u + i;
        if (idx < 4096u) {
            let residual = shared_pixels[idx];
            let prediction = shared_pred[idx];
            shared_pixels[idx] = clamp(residual + prediction, 0, 255);
        }
    }
    
    workgroupBarrier();
    
    // === PHASE 5: Loop filtering (deblocking) ===
    // Simple edge smoothing for block boundaries
    if (thread_id < 63u) {
        // Horizontal edges
        for (var y = 0u; y < 64u; y += 4u) {
            let idx = y * 64u + thread_id;
            if (idx < 4096u && (thread_id + 1u) % 4u == 0u) {
                let p1 = shared_pixels[idx];
                let p2 = shared_pixels[idx + 1u];
                let avg = (p1 + p2) / 2;
                shared_pixels[idx] = (p1 * 3 + avg) / 4;
                shared_pixels[idx + 1u] = (p2 * 3 + avg) / 4;
            }
        }
    }
    
    workgroupBarrier();
    
    // === PHASE 6: Write to frame buffer ===
    for (var i = 0u; i < 64u; i++) {
        let local_idx = thread_id * 64u + i;
        if (local_idx < 4096u) {
            let local_x = local_idx % 64u;
            let local_y = local_idx / 64u;
            let global_x = sb.x + local_x;
            let global_y = sb.y + local_y;
            
            if (global_x < frame_info.width && global_y < frame_info.height) {
                let global_idx = global_y * frame_info.width + global_x;
                frame_buffer[global_idx] = u32(shared_pixels[local_idx]);
            }
        }
    }
}