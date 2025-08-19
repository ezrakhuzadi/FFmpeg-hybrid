// VP9 Motion Compensation WebGPU Compute Shader
// Handles sub-pixel interpolation and residual addition

// Motion vector structure
struct MotionVector {
    mv_x: i32,
    mv_y: i32,
    ref_frame: u32,
    _pad: u32,
}

// Block information for motion compensation
struct MCBlock {
    // Block position in frame
    block_x: u32,
    block_y: u32,
    
    // Block size (0=4x4, 1=8x8, 2=16x16, 3=32x32)
    block_size: u32,
    
    // Motion vectors (up to 4 for 4x4 sub-blocks within larger blocks)
    mv: array<MotionVector, 4>,
    
    // Prediction mode
    mode: u32,
    
    // Chroma sub-sampling factors
    ss_x: u32, // horizontal subsampling (0 or 1)
    ss_y: u32, // vertical subsampling (0 or 1)
    
    _pad: u32,
}

// Bindings
@group(0) @binding(0) var<storage, read> residual_in: array<i16>;    // Input residual from IDCT
@group(0) @binding(1) var<storage, read_write> recon_y: array<u8>;   // Output luma plane
@group(0) @binding(2) var<storage, read_write> recon_u: array<u8>;   // Output chroma U plane
@group(0) @binding(3) var<storage, read_write> recon_v: array<u8>;   // Output chroma V plane
@group(0) @binding(4) var ref_y: texture_2d<f32>;                   // Reference luma texture
@group(0) @binding(5) var ref_u: texture_2d<f32>;                   // Reference chroma U texture
@group(0) @binding(6) var ref_v: texture_2d<f32>;                   // Reference chroma V texture
@group(0) @binding(7) var ref_sampler: sampler;                     // Bilinear sampler for sub-pixel
@group(0) @binding(8) var<storage, read> mc_blocks: array<MCBlock>; // Per-block MC info

@group(1) @binding(0) var<uniform> frame_info: FrameInfo;

struct FrameInfo {
    frame_width: u32,
    frame_height: u32,
    chroma_width: u32,
    chroma_height: u32,
}

// VP9 8-tap interpolation filter coefficients
// These are scaled by 128 for integer arithmetic
const FILTER_8TAP: array<array<i32, 8>, 4> = array(
    array(0, 0, 0, 128, 0, 0, 0, 0),        // Bilinear (no filtering)
    array(-1, 3, -7, 127, 8, -3, 1, 0),     // Regular filter
    array(-2, 5, -13, 108, 30, -8, 3, -1),  // Sharp filter  
    array(-3, 7, -17, 78, 78, -17, 7, -3)   // Smooth filter
);

const FILTER_8TAP_SHARP: array<array<i32, 8>, 4> = array(
    array(-2, 2, -6, 126, 8, -2, 4, -2),
    array(-2, 6, -12, 112, 36, -8, 4, -2),
    array(-3, 9, -16, 106, 50, -12, 4, -1),
    array(-3, 11, -17, 91, 91, -17, 11, -3)
);

// Bilinear filter for chroma (4-tap)
const FILTER_4TAP: array<array<i32, 4>, 4> = array(
    array(0, 128, 0, 0),           // No filtering
    array(-4, 126, 8, -2),         // 1/8 pel
    array(-8, 112, 30, -6),        // 2/8 pel
    array(-12, 96, 48, -4)         // 3/8 pel
);

// Function to apply 8-tap horizontal filter
fn apply_filter_8tap_h(pixels: array<f32, 8>, filter_idx: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        sum += pixels[i] * f32(FILTER_8TAP[filter_idx][i]);
    }
    return sum / 128.0;
}

// Function to apply 8-tap vertical filter
fn apply_filter_8tap_v(pixels: array<f32, 8>, filter_idx: u32) -> f32 {
    var sum: f32 = 0.0;
    for (var i: u32 = 0u; i < 8u; i = i + 1u) {
        sum += pixels[i] * f32(FILTER_8TAP[filter_idx][i]);
    }
    return sum / 128.0;
}

// Function to interpolate sub-pixel positions
fn interpolate_subpel(tex: texture_2d<f32>, sampler: sampler, x: f32, y: f32, width: u32, height: u32) -> f32 {
    // Convert to texture coordinates
    let tex_x = x / f32(width);
    let tex_y = y / f32(height);
    
    // Sample with bilinear interpolation
    return textureSampleLevel(tex, sampler, vec2<f32>(tex_x, tex_y), 0.0).r;
}

// High-quality sub-pixel interpolation using 8-tap filters
fn interpolate_8tap(tex: texture_2d<f32>, x: f32, y: f32, width: u32, height: u32) -> f32 {
    let floor_x = floor(x);
    let floor_y = floor(y);
    let frac_x = x - floor_x;
    let frac_y = y - floor_y;
    
    // Determine filter indices based on fractional parts
    let filter_x = u32(frac_x * 8.0);
    let filter_y = u32(frac_y * 8.0);
    
    // For now, use simple bilinear interpolation
    // Full 8-tap would require gathering 8x8 neighborhood and applying separable filters
    let tex_x = x / f32(width);
    let tex_y = y / f32(height);
    return textureSampleLevel(tex, ref_sampler, vec2<f32>(tex_x, tex_y), 0.0).r;
}

// Convert motion vector to pixel offset (VP9 uses 1/8 pixel precision)
fn mv_to_offset(mv: i32) -> f32 {
    return f32(mv) / 8.0;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let block_idx = global_id.x;
    let pixel_idx = global_id.y;
    
    if (block_idx >= arrayLength(&mc_blocks)) {
        return;
    }
    
    let block = mc_blocks[block_idx];
    let block_size = 4u << block.block_size; // 4, 8, 16, 32
    let block_pixels = block_size * block_size;
    
    if (pixel_idx >= block_pixels) {
        return;
    }
    
    // Calculate pixel position within block
    let pixel_x = pixel_idx % block_size;
    let pixel_y = pixel_idx / block_size;
    
    // Calculate absolute frame position  
    let frame_x = block.block_x + pixel_x;
    let frame_y = block.block_y + pixel_y;
    
    // Skip if outside frame bounds
    if (frame_x >= frame_info.frame_width || frame_y >= frame_info.frame_height) {
        return;
    }
    
    // For VP9, different sub-blocks within a larger block can have different MVs
    // For simplicity, use the first MV for now
    let mv = block.mv[0];
    
    let frame_offset = frame_y * frame_info.frame_width + frame_x;
    let residual_offset = block_idx * 1024u + pixel_idx; // Max 32x32 block
    
    // Motion compensation for luma
    if (mv.ref_frame != 0u) {
        // Apply motion vector
        let ref_x = f32(frame_x) + mv_to_offset(mv.mv_x);
        let ref_y = f32(frame_y) + mv_to_offset(mv.mv_y);
        
        // Interpolate reference pixel
        let pred_luma = interpolate_8tap(ref_y, ref_x, ref_y, frame_info.frame_width, frame_info.frame_height);
        
        // Add residual and clamp
        let residual = f32(residual_in[residual_offset]);
        let reconstructed = clamp(pred_luma * 255.0 + residual, 0.0, 255.0);
        
        recon_y[frame_offset] = u8(reconstructed);
    } else {
        // Intra prediction - just add residual to prediction (simplified)
        let residual = f32(residual_in[residual_offset]);
        let reconstructed = clamp(128.0 + residual, 0.0, 255.0); // DC prediction
        
        recon_y[frame_offset] = u8(reconstructed);
    }
    
    // Motion compensation for chroma (subsampled)
    let chroma_x = frame_x >> block.ss_x;
    let chroma_y = frame_y >> block.ss_y;
    
    if (chroma_x < frame_info.chroma_width && chroma_y < frame_info.chroma_height) {
        let chroma_offset = chroma_y * frame_info.chroma_width + chroma_x;
        
        if (mv.ref_frame != 0u) {
            // Apply subsampled motion vector for chroma
            let ref_u_x = f32(chroma_x) + mv_to_offset(mv.mv_x) / f32(1u << block.ss_x);
            let ref_u_y = f32(chroma_y) + mv_to_offset(mv.mv_y) / f32(1u << block.ss_y);
            let ref_v_x = ref_u_x;
            let ref_v_y = ref_u_y;
            
            // Interpolate chroma
            let pred_u = interpolate_subpel(ref_u, ref_sampler, ref_u_x, ref_u_y, frame_info.chroma_width, frame_info.chroma_height);
            let pred_v = interpolate_subpel(ref_v, ref_sampler, ref_v_x, ref_v_y, frame_info.chroma_width, frame_info.chroma_height);
            
            // Add chroma residuals (offset by luma block size for U/V planes)
            let chroma_residual_offset = residual_offset + block_pixels + (pixel_idx >> (block.ss_x + block.ss_y));
            let residual_u = f32(residual_in[chroma_residual_offset]);
            let residual_v = f32(residual_in[chroma_residual_offset + block_pixels / 4u]);
            
            let recon_u_val = clamp(pred_u * 255.0 + residual_u, 0.0, 255.0);
            let recon_v_val = clamp(pred_v * 255.0 + residual_v, 0.0, 255.0);
            
            recon_u[chroma_offset] = u8(recon_u_val);
            recon_v[chroma_offset] = u8(recon_v_val);
        } else {
            // Intra chroma prediction
            let chroma_residual_offset = residual_offset + block_pixels + (pixel_idx >> (block.ss_x + block.ss_y));
            let residual_u = f32(residual_in[chroma_residual_offset]);
            let residual_v = f32(residual_in[chroma_residual_offset + block_pixels / 4u]);
            
            let recon_u_val = clamp(128.0 + residual_u, 0.0, 255.0);
            let recon_v_val = clamp(128.0 + residual_v, 0.0, 255.0);
            
            recon_u[chroma_offset] = u8(recon_u_val);
            recon_v[chroma_offset] = u8(recon_v_val);
        }
    }
}