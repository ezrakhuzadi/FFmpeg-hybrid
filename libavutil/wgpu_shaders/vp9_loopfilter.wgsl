// VP9 Loop Filter WebGPU Compute Shader
// Applies deblocking and deringing filters

// Loop filter parameters for each 8x8 block
struct LoopFilterParams {
    // Filter levels for each edge (left, top, right, bottom)
    level: array<u32, 4>,
    
    // Sharpness parameter
    sharpness: u32,
    
    // Filter type (0=normal, 1=simple)
    filter_type: u32,
    
    // Whether this block is at frame boundary
    boundary_flags: u32, // bit 0=left, bit 1=top, bit 2=right, bit 3=bottom
    
    _pad: u32,
}

// Bindings
@group(0) @binding(0) var<storage, read_write> frame_y: array<u8>;   // Luma plane (read/write for in-place filtering)
@group(0) @binding(1) var<storage, read_write> frame_u: array<u8>;   // Chroma U plane
@group(0) @binding(2) var<storage, read_write> frame_v: array<u8>;   // Chroma V plane
@group(0) @binding(3) var<storage, read> filter_params: array<LoopFilterParams>; // Per-block filter parameters

@group(1) @binding(0) var<uniform> frame_info: FrameInfo;

struct FrameInfo {
    frame_width: u32,
    frame_height: u32,
    chroma_width: u32,
    chroma_height: u32,
    blocks_x: u32,  // Number of 8x8 blocks horizontally
    blocks_y: u32,  // Number of 8x8 blocks vertically
}

// Loop filter threshold tables
const LOOP_FILTER_THRESH: array<u32, 64> = array(
    0u, 1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 3u, 3u, 3u, 3u, 4u, 4u, 4u,
    4u, 5u, 5u, 5u, 5u, 6u, 6u, 6u, 6u, 7u, 7u, 7u, 7u, 8u, 8u, 8u,
    8u, 9u, 9u, 9u, 9u, 10u, 10u, 10u, 10u, 11u, 11u, 11u, 11u, 12u, 12u, 12u,
    12u, 13u, 13u, 13u, 13u, 14u, 14u, 14u, 14u, 15u, 15u, 15u, 15u, 16u, 16u, 16u
);

const LOOP_FILTER_LIMIT: array<u32, 64> = array(
    30u, 25u, 20u, 20u, 15u, 15u, 14u, 14u, 13u, 13u, 12u, 12u, 11u, 11u, 10u, 10u,
    9u, 9u, 8u, 8u, 7u, 7u, 7u, 7u, 6u, 6u, 6u, 6u, 5u, 5u, 5u, 5u,
    4u, 4u, 4u, 4u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 3u, 2u, 2u, 2u, 2u,
    2u, 2u, 2u, 2u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u
);

// VP9 loop filter mask calculation
fn compute_filter_mask(p3: u32, p2: u32, p1: u32, p0: u32, q0: u32, q1: u32, q2: u32, q3: u32,
                      limit: u32, blimit: u32, thresh: u32) -> bool {
    let mask1 = abs(i32(p1) - i32(p0)) <= i32(thresh);
    let mask2 = abs(i32(q1) - i32(q0)) <= i32(thresh);
    let mask3 = abs(i32(p0) - i32(q0)) * 2 + abs(i32(p1) - i32(q1)) / 2 <= i32(blimit);
    let mask4 = abs(i32(p3) - i32(p2)) <= i32(limit);
    let mask5 = abs(i32(p2) - i32(p1)) <= i32(limit);
    let mask6 = abs(i32(q2) - i32(q1)) <= i32(limit);
    let mask7 = abs(i32(q3) - i32(q2)) <= i32(limit);
    
    return mask1 && mask2 && mask3 && mask4 && mask5 && mask6 && mask7;
}

// VP9 high edge variance test
fn is_high_edge_var(p1: u32, p0: u32, q0: u32, q1: u32, thresh: u32) -> bool {
    return (abs(i32(p1) - i32(p0)) > i32(thresh)) || (abs(i32(q1) - i32(q0)) > i32(thresh));
}

// VP9 4-tap loop filter (for normal filtering)
fn filter_4tap(p1: u32, p0: u32, q0: u32, q1: u32) -> vec2<u32> {
    let a = 3 * (i32(q0) - i32(p0)) + 4 * (i32(p1) - i32(q1));
    let filter1 = clamp((a + 4) >> 3, -9, 9);
    let filter2 = clamp((a + 3) >> 3, -9, 9);
    
    let new_p0 = clamp(i32(p0) + filter2, 0, 255);
    let new_q0 = clamp(i32(q0) - filter1, 0, 255);
    
    return vec2<u32>(u32(new_p0), u32(new_q0));
}

// VP9 6-tap loop filter (for wide filtering)
fn filter_6tap(p2: u32, p1: u32, p0: u32, q0: u32, q1: u32, q2: u32) -> vec4<u32> {
    let a = 3 * (i32(q0) - i32(p0)) + 4 * (i32(p1) - i32(q1));
    let filter1 = clamp((a + 4) >> 3, -9, 9);
    let filter2 = clamp((a + 3) >> 3, -9, 9);
    
    // Apply different strengths to different pixels
    let adj1 = filter1;
    let adj0 = filter2;
    let adj_p2 = adj1 >> 1;
    let adj_q2 = adj0 >> 1;
    
    let new_p2 = clamp(i32(p2) + adj_p2, 0, 255);
    let new_p1 = clamp(i32(p1) + adj1, 0, 255);
    let new_p0 = clamp(i32(p0) + adj0, 0, 255);
    let new_q0 = clamp(i32(q0) - adj0, 0, 255);
    let new_q1 = clamp(i32(q1) - adj1, 0, 255);
    let new_q2 = clamp(i32(q2) - adj_q2, 0, 255);
    
    return vec4<u32>(u32(new_p1), u32(new_p0), u32(new_q0), u32(new_q1));
}

// Apply horizontal edge filter (filtering vertically across horizontal edge)
fn filter_horizontal_edge(frame: ptr<storage, array<u8>, read_write>, 
                         width: u32, height: u32,
                         x: u32, y: u32, 
                         level: u32) {
    if (level == 0u || y < 4u || y >= height - 4u) {
        return;
    }
    
    let stride = width;
    let base_offset = y * stride + x;
    
    // Load 8 pixels: p3, p2, p1, p0 | q0, q1, q2, q3
    let p3 = u32((*frame)[base_offset - 4u * stride]);
    let p2 = u32((*frame)[base_offset - 3u * stride]);
    let p1 = u32((*frame)[base_offset - 2u * stride]);
    let p0 = u32((*frame)[base_offset - stride]);
    let q0 = u32((*frame)[base_offset]);
    let q1 = u32((*frame)[base_offset + stride]);
    let q2 = u32((*frame)[base_offset + 2u * stride]);
    let q3 = u32((*frame)[base_offset + 3u * stride]);
    
    let limit = LOOP_FILTER_LIMIT[level];
    let blimit = limit * 2u + 20u;
    let thresh = LOOP_FILTER_THRESH[level];
    
    // Check if filtering should be applied
    if (!compute_filter_mask(p3, p2, p1, p0, q0, q1, q2, q3, limit, blimit, thresh)) {
        return;
    }
    
    // Determine filter width based on edge variance
    if (is_high_edge_var(p1, p0, q0, q1, thresh)) {
        // Use 4-tap filter for high variance edges
        let filtered = filter_4tap(p1, p0, q0, q1);
        (*frame)[base_offset - stride] = u8(filtered.x);
        (*frame)[base_offset] = u8(filtered.y);
    } else {
        // Use 6-tap filter for smooth edges
        let filtered = filter_6tap(p2, p1, p0, q0, q1, q2);
        (*frame)[base_offset - 2u * stride] = u8(filtered.x);
        (*frame)[base_offset - stride] = u8(filtered.y);
        (*frame)[base_offset] = u8(filtered.z);
        (*frame)[base_offset + stride] = u8(filtered.w);
    }
}

// Apply vertical edge filter (filtering horizontally across vertical edge)
fn filter_vertical_edge(frame: ptr<storage, array<u8>, read_write>,
                       width: u32, height: u32,
                       x: u32, y: u32,
                       level: u32) {
    if (level == 0u || x < 4u || x >= width - 4u) {
        return;
    }
    
    let stride = width;
    let base_offset = y * stride + x;
    
    // Load 8 pixels: p3, p2, p1, p0 | q0, q1, q2, q3
    let p3 = u32((*frame)[base_offset - 4u]);
    let p2 = u32((*frame)[base_offset - 3u]);
    let p1 = u32((*frame)[base_offset - 2u]);
    let p0 = u32((*frame)[base_offset - 1u]);
    let q0 = u32((*frame)[base_offset]);
    let q1 = u32((*frame)[base_offset + 1u]);
    let q2 = u32((*frame)[base_offset + 2u]);
    let q3 = u32((*frame)[base_offset + 3u]);
    
    let limit = LOOP_FILTER_LIMIT[level];
    let blimit = limit * 2u + 20u;
    let thresh = LOOP_FILTER_THRESH[level];
    
    // Check if filtering should be applied
    if (!compute_filter_mask(p3, p2, p1, p0, q0, q1, q2, q3, limit, blimit, thresh)) {
        return;
    }
    
    // Determine filter width based on edge variance
    if (is_high_edge_var(p1, p0, q0, q1, thresh)) {
        // Use 4-tap filter for high variance edges
        let filtered = filter_4tap(p1, p0, q0, q1);
        (*frame)[base_offset - 1u] = u8(filtered.x);
        (*frame)[base_offset] = u8(filtered.y);
    } else {
        // Use 6-tap filter for smooth edges
        let filtered = filter_6tap(p2, p1, p0, q0, q1, q2);
        (*frame)[base_offset - 2u] = u8(filtered.x);
        (*frame)[base_offset - 1u] = u8(filtered.y);
        (*frame)[base_offset] = u8(filtered.z);
        (*frame)[base_offset + 1u] = u8(filtered.w);
    }
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_x = global_id.x;
    let block_y = global_id.y;
    
    if (block_x >= frame_info.blocks_x || block_y >= frame_info.blocks_y) {
        return;
    }
    
    let block_idx = block_y * frame_info.blocks_x + block_x;
    if (block_idx >= arrayLength(&filter_params)) {
        return;
    }
    
    let params = filter_params[block_idx];
    
    // Convert block coordinates to pixel coordinates (8x8 blocks)
    let pixel_x = block_x * 8u;
    let pixel_y = block_y * 8u;
    
    // Apply vertical edge filtering (left and right edges of block)
    if ((params.boundary_flags & 1u) == 0u) { // Not at left frame boundary
        for (var y: u32 = 0u; y < 8u; y = y + 1u) {
            filter_vertical_edge(&frame_y, frame_info.frame_width, frame_info.frame_height,
                                pixel_x, pixel_y + y, params.level[0]);
        }
    }
    
    // Apply horizontal edge filtering (top and bottom edges of block)
    if ((params.boundary_flags & 2u) == 0u) { // Not at top frame boundary
        for (var x: u32 = 0u; x < 8u; x = x + 1u) {
            filter_horizontal_edge(&frame_y, frame_info.frame_width, frame_info.frame_height,
                                  pixel_x + x, pixel_y, params.level[1]);
        }
    }
    
    // Apply chroma filtering (subsampled)
    let chroma_x = pixel_x / 2u;
    let chroma_y = pixel_y / 2u;
    
    if (chroma_x < frame_info.chroma_width && chroma_y < frame_info.chroma_height) {
        // Apply to U plane
        if ((params.boundary_flags & 1u) == 0u) {
            for (var y: u32 = 0u; y < 4u; y = y + 1u) {
                filter_vertical_edge(&frame_u, frame_info.chroma_width, frame_info.chroma_height,
                                    chroma_x, chroma_y + y, params.level[0] >> 1u);
            }
        }
        
        if ((params.boundary_flags & 2u) == 0u) {
            for (var x: u32 = 0u; x < 4u; x = x + 1u) {
                filter_horizontal_edge(&frame_u, frame_info.chroma_width, frame_info.chroma_height,
                                      chroma_x + x, chroma_y, params.level[1] >> 1u);
            }
        }
        
        // Apply to V plane
        if ((params.boundary_flags & 1u) == 0u) {
            for (var y: u32 = 0u; y < 4u; y = y + 1u) {
                filter_vertical_edge(&frame_v, frame_info.chroma_width, frame_info.chroma_height,
                                    chroma_x, chroma_y + y, params.level[0] >> 1u);
            }
        }
        
        if ((params.boundary_flags & 2u) == 0u) {
            for (var x: u32 = 0u; x < 4u; x = x + 1u) {
                filter_horizontal_edge(&frame_v, frame_info.chroma_width, frame_info.chroma_height,
                                      chroma_x + x, chroma_y, params.level[1] >> 1u);
            }
        }
    }
}