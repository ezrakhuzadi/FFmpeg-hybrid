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
#include <stdatomic.h>

#include "libavutil/buffer.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_webgpu.h"

#include "avcodec.h"
#include "vp9shared.h"

// Forward declarations
typedef struct VP9Context VP9Context;
typedef struct VP9SuperblockGPU VP9SuperblockGPU;

// Zero-copy buffer system for unified memory
typedef struct {
    WGPUBuffer persistent_mapped;     // Always mapped, never unmap
    void *cpu_ptr;                    // Direct CPU write pointer
    WGPUBuffer gpu_storage;           // GPU reads from here
    size_t size;
    int is_mapped;
} ZeroCopyBuffer;

// Structure-of-Arrays for GPU coalesced access (128-byte aligned)
typedef struct __attribute__((aligned(128))) {
    // Pack data for optimal GPU memory access
    int16_t all_coefficients[8192][1024];  // All coeffs together (8K max SBs)
    uint32_t all_modes[8192];              // All modes together  
    int16_t all_mvs[8192][2];              // All motion vectors together
    uint32_t partition_masks[8192];        // All partitions together
    uint32_t transform_masks[8192];        // All transform masks
    uint32_t sb_x_coords[8192];            // X coordinates
    uint32_t sb_y_coords[8192];            // Y coordinates
    uint32_t sb_count;                     // Active superblocks
} SOA_SuperblockData;

// Lockless ring buffer for CPU-GPU communication
typedef struct {
    struct VP9SuperblockGPU *ring_buffer;  // Dynamic allocation
    _Atomic uint64_t write_index;          // CPU writes here
    _Atomic uint64_t read_index;           // GPU reads from here
    char padding[64];                      // Cache line separation
} LocklessRingBuffer;

// Dynamic workgroup configuration
typedef struct {
    uint32_t vendor_optimal_size;     // 32 Intel, 64 AMD, 128 NVIDIA
    uint32_t resolution_multiplier;   // Larger for higher res
    uint32_t memory_bandwidth_factor; // Adjust based on memory speed
} DynamicWorkgroupConfig;

// Transform block metadata (matches WGSL struct)
typedef struct VP9WebGPUTransformBlock {
    uint32_t block_x, block_y;
    uint32_t transform_size;
    uint32_t transform_type_x, transform_type_y;
    uint32_t qindex;
    uint32_t _pad[2];
} VP9WebGPUTransformBlock;

// Superblock metadata for GPU processing - matches Intel's approach
typedef struct VP9SuperblockGPU {
    int16_t coeffs[64*64];         // Entire superblock coefficients
    int32_t x, y;                  // Superblock position in pixels
    int32_t mode;                  // Prediction mode
    int16_t motion_vectors[2];     // Motion vectors for MC
    uint32_t partition_mask;       // How the superblock is partitioned
    uint32_t transform_mask;       // Which transforms are present
} VP9SuperblockGPU;

// Transform within a superblock
typedef struct VP9WebGPUSuperblockTransform {
    uint32_t local_x, local_y;     // Position within superblock (0-63)
    uint32_t tx_size;              // Transform size
    uint32_t tx_type;              // Transform type
    uint32_t coeff_offset;         // Offset into coefficient buffer
    uint32_t eob;                  // End of block
    uint32_t _pad[2];
} VP9WebGPUSuperblockTransform;

// Triple-buffered frame pipeline
typedef struct {
    ZeroCopyBuffer frame_buffers[3];
    WGPUCommandBuffer pending_commands[3];
    _Atomic int cpu_frame_idx;     // CPU working on this frame
    _Atomic int gpu_frame_idx;     // GPU working on this frame
    _Atomic int display_frame_idx; // Ready for display
} TripleBufferPipeline;

// Universal pre-compiled pipeline cache
typedef enum {
    RESOLUTION_1080P, RESOLUTION_4K, RESOLUTION_8K,
    GPU_INTEGRATED, GPU_DISCRETE_MID, GPU_DISCRETE_HIGH,
    MEMORY_UNIFIED, MEMORY_DISCRETE
} ShaderVariant;

typedef struct {
    WGPUComputePipeline pipelines[3][3][2]; // [resolution][gpu_tier][memory_type]
    uint32_t optimal_workgroup_sizes[3][3][2];
    uint32_t optimal_batch_sizes[3][3][2];
} UniversalPipelineCache;

// Intelligent reference frame cache
typedef struct {
    WGPUTexture reference_frames[8];      // VP9 allows up to 8 refs
    WGPUTexture prediction_cache[64];     // Cache common prediction blocks
    uint64_t frame_usage_stats[8];        // Track which refs are used most
    uint32_t prediction_hit_count[64];    // Track prediction cache hits
} IntelligentFrameCache;

typedef struct VP9WebGPUContext {
    AVBufferRef *device_ref;
    AVWebGPUDeviceContext *device_ctx;
    
    // NEW: Zero-copy buffer system
    ZeroCopyBuffer *persistent_buffers;
    SOA_SuperblockData *soa_data;
    LocklessRingBuffer *ring_buffer;
    
    // NEW: Triple-buffered pipeline
    TripleBufferPipeline triple_buffer;
    
    // NEW: Pre-compiled shader cache
    UniversalPipelineCache shader_cache;
    
    // NEW: Reference frame cache
    IntelligentFrameCache ref_cache;
    
    // NEW: Dynamic workgroup config
    DynamicWorkgroupConfig workgroup_config;
    
    // NEW: Mega kernel for complete decode
    WGPUComputePipeline mega_kernel_pipeline;
    
    // WebGPU compute pipelines (legacy - will be replaced by mega kernel)
    WGPUComputePipeline idct4x4_pipeline;
    WGPUComputePipeline idct8x8_pipeline;
    WGPUComputePipeline idct16x16_pipeline;
    WGPUComputePipeline idct32x32_pipeline;
    WGPUComputePipeline mc_pipeline;
    WGPUComputePipeline loopfilter_pipeline;
    WGPUComputePipeline superblock_pipeline;  // New superblock processing pipeline
    
    // Bind group layouts
    WGPUBindGroupLayout idct_bind_group_layout;
    WGPUBindGroupLayout mc_bind_group_layout;
    WGPUBindGroupLayout loopfilter_bind_group_layout;
    WGPUBindGroupLayout superblock_bind_group_layout;
    
    // Persistent buffers for metadata
    WGPUBuffer transform_blocks_buffer;
    WGPUBuffer mc_blocks_buffer;
    WGPUBuffer loopfilter_blocks_buffer;
    WGPUBuffer dequant_table_buffer;
    
    // Frame info uniform buffer
    WGPUBuffer frame_info_buffer;
    
    // MEGA KERNEL persistent resources
    WGPUBuffer mega_uniform_buffer;      // Frame info for mega kernel
    WGPUBuffer mega_superblock_buffer;   // Superblock data buffer
    WGPUBuffer mega_frame_buffer;        // Output frame buffer
    WGPUTexture mega_ref_texture;        // Reference frame texture
    WGPUTextureView mega_ref_view;       // Reference frame view
    WGPUSampler mega_sampler;            // Texture sampler
    WGPUBindGroup mega_bind_group;       // Persistent bind group
    int mega_resources_initialized;      // Whether resources are created
    
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
    
    // Tile-level superblock batching (Intel-style)
    struct {
        VP9SuperblockGPU *superblocks;
        int num_superblocks;
        int capacity_sb;
        
        // Current superblock being accumulated
        VP9SuperblockGPU current_sb;
        int current_sb_x, current_sb_y;
        int has_coeffs;  // Whether current SB has any non-zero coeffs
        
        // Storage for all accumulated superblocks during frame decode
        VP9SuperblockGPU *accumulated_sbs;
        int accumulated_count;
        int accumulated_capacity;
        
        // Direct mapped GPU buffer for zero-copy accumulation
        WGPUBuffer direct_map_buffer;
        VP9SuperblockGPU *mapped_sbs;  // Direct pointer to GPU memory
        int is_mapped;
        
        // Zero-copy ring buffers (two-buffer strategy)
        struct {
            WGPUBuffer mappable;    // CPU writes here (MapWrite | CopySrc)
            WGPUBuffer storage;     // GPU reads here (Storage | CopyDst)
            void *mapped_ptr;       // Mapped pointer for zero-copy writes
            int is_mapped;
        } ring_buffers[3];  // Triple buffering
        
        int write_index;
        int read_index;
        
        // Synchronization
        pthread_mutex_t ring_mutex;
        pthread_cond_t gpu_ready;
        pthread_cond_t cpu_ready;
    } tile_batch;
    
    // GPU buffers for superblock processing
    WGPUBuffer sb_info_buffer;
    WGPUBuffer sb_transform_buffer;
    WGPUBuffer sb_coeff_buffer;
    WGPUBuffer frame_buffer;
    WGPUBuffer prediction_buffer;
    
    // Motion compensation batching
    struct {
        struct VP9WebGPUMCBlock *blocks;
        int count;
        int capacity;
    } mc_batch;
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

// Add MC block to batch for later execution
int ff_vp9_webgpu_add_mc_block(VP9WebGPUContext *ctx, const VP9WebGPUMCBlock *block);

// Execute all batched MC blocks
int ff_vp9_webgpu_flush_mc_batch(VP9WebGPUContext *ctx, VP9Context *s);

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

// Process entire tile on GPU (Intel-style)
int ff_vp9_webgpu_process_tile(VP9WebGPUContext *ctx, VP9Context *s,
                               int tile_row, int tile_col);

// Apply GPU results to frame
int ff_vp9_webgpu_apply_tile_results(VP9WebGPUContext *ctx, VP9Context *s,
                                     const void *gpu_data, size_t data_size);

// Submit tile batch for GPU processing
int ff_vp9_webgpu_submit_tile_batch(VP9WebGPUContext *ctx, 
                                    VP9SuperblockGPU *superblocks, 
                                    int sb_count, VP9Context *s);

// Submit tile batch directly from mapped buffer (zero-copy)
int ff_vp9_webgpu_submit_tile_batch_direct(VP9WebGPUContext *ctx,
                                           WGPUBuffer mapped_buffer,
                                           int sb_count, VP9Context *s);

// Extract superblock data for GPU
void ff_vp9_webgpu_extract_superblock_data(VP9Context *s, 
                                           int sb_x, int sb_y,
                                           VP9SuperblockGPU *sb_gpu);

// Accumulate block coefficients into current superblock
void ff_vp9_webgpu_accumulate_block_coeffs(VP9WebGPUContext *ctx,
                                           int block_x, int block_y,
                                           int16_t *coeffs, int size,
                                           int tx, int plane);

// Accumulate motion compensation data for inter blocks
void ff_vp9_webgpu_accumulate_motion_data(VP9WebGPUContext *ctx,
                                          int sb_x, int sb_y,
                                          const VP9mv *mv0, const VP9mv *mv1,
                                          int ref0, int ref1,
                                          int comp, int mode, int bs);

// Accumulate intra prediction data
void ff_vp9_webgpu_accumulate_intra_data(VP9WebGPUContext *ctx,
                                         int sb_x, int sb_y,
                                         int mode, int bs);

// Begin frame decode with mapped GPU buffer for zero-copy
int ff_vp9_webgpu_begin_frame(VP9WebGPUContext *ctx);

// End frame decode and submit to GPU
int ff_vp9_webgpu_end_frame(VP9WebGPUContext *ctx, VP9Context *s);

// NEW: Zero-copy functions
int ff_vp9_webgpu_init_zero_copy_buffers(VP9WebGPUContext *ctx, int width, int height);
void ff_vp9_webgpu_destroy_zero_copy_buffers(VP9WebGPUContext *ctx);
void ff_vp9_webgpu_decode_coefficients_zero_copy(VP9Context *s, ZeroCopyBuffer *buf);

// NEW: Lockless ring buffer functions
void ff_vp9_webgpu_cpu_write_coefficients(LocklessRingBuffer *ring, VP9SuperblockGPU *data);
uint32_t ff_vp9_webgpu_gpu_read_superblock_batch(LocklessRingBuffer *ring, VP9SuperblockGPU *batch, uint32_t max_batch_size);

// NEW: Dynamic optimization functions
uint32_t ff_vp9_webgpu_calculate_optimal_workgroup_size(VP9WebGPUContext *ctx, int width, int height);
uint32_t ff_vp9_webgpu_calculate_optimal_batch_size(VP9WebGPUContext *ctx, int width, int height);

// NEW: Pre-compiled shader selection
WGPUComputePipeline ff_vp9_webgpu_select_optimal_pipeline(UniversalPipelineCache *cache, int width, int height);

// NEW: Frame-level mega dispatch
void ff_vp9_webgpu_dispatch_complete_frame(VP9WebGPUContext *ctx, VP9Context *s);

// NEW: Triple-buffered pipeline
void ff_vp9_webgpu_async_decode_pipeline(VP9WebGPUContext *ctx, VP9Context *s);

// NEW: Reference frame caching
void ff_vp9_webgpu_update_reference_cache_strategy(IntelligentFrameCache *cache, VP9Context *s);

#endif /* AVCODEC_VP9_WEBGPU_H */