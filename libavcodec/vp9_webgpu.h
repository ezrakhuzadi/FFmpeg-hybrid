/*
 * VP9 WebGPU acceleration
 *
 * Copyright (C) 2024 FFmpeg contributors
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

#ifndef AVCODEC_VP9_WEBGPU_H
#define AVCODEC_VP9_WEBGPU_H

#include <webgpu.h>
#include <pthread.h>

#include "libavutil/buffer.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_webgpu.h"

#include "avcodec.h"

// Forward declarations
typedef struct VP9Context VP9Context;

// Transform block metadata (matches WGSL struct)
typedef struct VP9WebGPUTransformBlock {
    uint32_t block_x, block_y;
    uint32_t transform_size;
    uint32_t transform_type_x, transform_type_y;
    uint32_t qindex;
    uint32_t _pad[2];
} VP9WebGPUTransformBlock;

typedef struct VP9WebGPUContext {
    AVBufferRef *device_ref;
    AVWebGPUDeviceContext *device_ctx;
    
    // WebGPU compute pipelines
    WGPUComputePipeline idct4x4_pipeline;
    WGPUComputePipeline idct8x8_pipeline;
    WGPUComputePipeline idct16x16_pipeline;
    WGPUComputePipeline idct32x32_pipeline;
    WGPUComputePipeline mc_pipeline;
    WGPUComputePipeline loopfilter_pipeline;
    
    // Bind group layouts
    WGPUBindGroupLayout idct_bind_group_layout;
    WGPUBindGroupLayout mc_bind_group_layout;
    WGPUBindGroupLayout loopfilter_bind_group_layout;
    
    // Persistent buffers for metadata
    WGPUBuffer transform_blocks_buffer;
    WGPUBuffer mc_blocks_buffer;
    WGPUBuffer loopfilter_blocks_buffer;
    WGPUBuffer dequant_table_buffer;
    
    // Frame info uniform buffer
    WGPUBuffer frame_info_buffer;
    
    // Working buffers (allocated per frame)
    WGPUBuffer coefficients_buffer;
    WGPUBuffer residuals_buffer;
    // Output buffer for motion compensation (luma plane)
    WGPUBuffer mc_output_buffer;
    
    // Reference frame textures (for motion compensation)
    WGPUTexture ref_textures_y[8];  // VP9 supports up to 8 reference frames
    WGPUTexture ref_textures_u[8];
    WGPUTexture ref_textures_v[8];
    WGPUTextureView ref_texture_views_y[8];
    WGPUTextureView ref_texture_views_u[8];
    WGPUTextureView ref_texture_views_v[8];
    
    // Sampler for interpolation
    WGPUSampler bilinear_sampler;
    
    // Current frame dimensions
    int frame_width, frame_height;
    int chroma_width, chroma_height;
    
    // VP9 profile and bit depth support
    int profile;      // VP9 profile (0-3)
    int bit_depth;    // 8, 10, or 12 bits
    int subsampling_x; // Chroma subsampling
    int subsampling_y;
    
    // Number of blocks for current frame
    int num_transform_blocks;
    int num_mc_blocks;
    int num_loopfilter_blocks;
    
    // Performance optimization: buffer pools
    struct {
        WGPUBuffer buffers[4];  // Pool of reusable buffers
        int sizes[4];          // Size of each buffer
        int in_use[4];         // Whether buffer is in use
    } buffer_pool;
    
    // Batching for reduced overhead
    int batch_size;            // Number of blocks to process per dispatch
    int max_batch_size;        // Maximum batch size (tunable)
    
    // Batch accumulation buffers
    struct {
        VP9WebGPUTransformBlock *blocks_4x4;
        VP9WebGPUTransformBlock *blocks_8x8;
        VP9WebGPUTransformBlock *blocks_16x16;
        VP9WebGPUTransformBlock *blocks_32x32;
        int16_t *coeffs_4x4;
        int16_t *coeffs_8x8;
        int16_t *coeffs_16x16;
        int16_t *coeffs_32x32;
        // Store destination pointers and strides for writeback
        struct {
            uint8_t *dst;
            ptrdiff_t stride;
        } *dests_4x4, *dests_8x8, *dests_16x16, *dests_32x32;
        int count_4x4;
        int count_8x8;
        int count_16x16;
        int count_32x32;
        int capacity;  // Max blocks per batch
    } transform_batch;
    
    // Persistent GPU buffers for batching
    WGPUBuffer batch_metadata_buffer;
    WGPUBuffer batch_coeffs_buffer;
    WGPUBuffer batch_output_buffer;
    
    // Command encoder for batching
    WGPUCommandEncoder batch_encoder;
    int encoder_active;
    
    // Thread safety
    pthread_mutex_t mutex;
    pthread_cond_t cond;
    int processing_count;  // Number of threads currently processing
    
    // Per-thread contexts for multi-threading
    struct {
        WGPUCommandEncoder encoder;
        WGPUBuffer staging_buffer;
        int active;
    } thread_contexts[16];  // Support up to 16 threads
    int num_threads;
} VP9WebGPUContext;

// Motion compensation block metadata
typedef struct VP9WebGPUMCBlock {
    uint32_t block_x, block_y;
    uint32_t block_size;
    struct {
        int32_t mv_x, mv_y;
        uint32_t ref_frame;
        uint32_t _pad;
    } mv[4];
    uint32_t mode;
    uint32_t ss_x, ss_y;
    uint32_t _pad;
} VP9WebGPUMCBlock;

// Loop filter block metadata  
typedef struct VP9WebGPULoopFilterBlock {
    uint32_t level[4];  // left, top, right, bottom
    uint32_t sharpness;
    uint32_t filter_type;
    uint32_t boundary_flags;
    uint32_t _pad;
} VP9WebGPULoopFilterBlock;

// Frame info uniform data
typedef struct VP9WebGPUFrameInfo {
    uint32_t frame_width;
    uint32_t frame_height;
    uint32_t chroma_width;
    uint32_t chroma_height;
    uint32_t bit_depth;      // 8, 10, or 12
    uint32_t profile;        // VP9 profile (0-3)
    uint32_t subsampling_x;  // Chroma subsampling
    uint32_t subsampling_y;
} VP9WebGPUFrameInfo;

// Initialize VP9 WebGPU acceleration context
int ff_vp9_webgpu_init(AVCodecContext *avctx, VP9WebGPUContext **ctx);

// Uninitialize and free VP9 WebGPU context
void ff_vp9_webgpu_uninit(VP9WebGPUContext **ctx);

// Set up reference frames for motion compensation
int ff_vp9_webgpu_setup_references(VP9WebGPUContext *ctx, VP9Context *s);

// Execute inverse transform on WebGPU
int ff_vp9_webgpu_transform(VP9WebGPUContext *ctx, VP9Context *s, 
                           const int16_t *coeffs, int num_blocks,
                           const VP9WebGPUTransformBlock *block_info);

// Execute motion compensation on WebGPU
int ff_vp9_webgpu_motion_compensation(VP9WebGPUContext *ctx, VP9Context *s,
                                     const VP9WebGPUMCBlock *block_info,
                                     int num_blocks);

// Execute loop filtering on WebGPU
int ff_vp9_webgpu_loop_filter(VP9WebGPUContext *ctx, VP9Context *s,
                              const VP9WebGPULoopFilterBlock *filter_info,
                              int num_blocks);

// Create and update dequantization tables
int ff_vp9_webgpu_update_dequant_tables(VP9WebGPUContext *ctx, VP9Context *s);

// Simple inverse transform function for single block (used by vp9recon.c)
int ff_vp9_webgpu_inverse_transform(VP9WebGPUContext *ctx, uint8_t *dst, ptrdiff_t stride,
                                   int16_t *coeffs, int eob, int tx);

// Plane-aware inverse transform (0=Y, 1=U, 2=V)
int ff_vp9_webgpu_inverse_transform_plane(VP9WebGPUContext *ctx, uint8_t *dst, ptrdiff_t stride,
                                          int16_t *coeffs, int eob, int tx, int plane);

// Full transform with type support (DCT_DCT, DCT_ADST, ADST_DCT, ADST_ADST)
int ff_vp9_webgpu_inverse_transform_type(VP9WebGPUContext *ctx, uint8_t *dst, ptrdiff_t stride,
                                         int16_t *coeffs, int eob, int tx, int txtp, int plane);

// Batched transform operations
int ff_vp9_webgpu_begin_batch(VP9WebGPUContext *ctx);
int ff_vp9_webgpu_add_transform_to_batch(VP9WebGPUContext *ctx, 
                                         uint32_t block_x, uint32_t block_y,
                                         int16_t *coeffs, int eob, 
                                         int tx, int txtp);
int ff_vp9_webgpu_flush_batch(VP9WebGPUContext *ctx, VP9Context *s);

// Execute batched transforms by size
int ff_vp9_webgpu_execute_transform_batch(VP9WebGPUContext *ctx, 
                                          int tx_size,
                                          VP9WebGPUTransformBlock *blocks,
                                          int16_t *coeffs,
                                          int num_blocks);

// Process entire tile row on GPU
int ff_vp9_webgpu_process_tile_row(VP9WebGPUContext *ctx, VP9Context *s,
                                   uint8_t *dst_y, uint8_t *dst_u, uint8_t *dst_v,
                                   int row_start, int row_end);

#endif /* AVCODEC_VP9_WEBGPU_H */