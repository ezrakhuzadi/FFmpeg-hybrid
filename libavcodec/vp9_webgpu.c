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

#include "config.h"

#include <webgpu.h>
#include <string.h>

#include "libavutil/buffer.h"
#include "libavutil/common.h"
#include "libavutil/hwcontext.h"
#include "libavutil/hwcontext_webgpu.h"
#include "libavutil/log.h"
#include "libavutil/mem.h"
#include "libavutil/time.h"

#include "avcodec.h"
#include "vp9_webgpu.h"
#include "vp9_webgpu_shaders.h"
#include "vp9dec.h"
#include "vp9data.h"
#include "vp9.h"

// Structure for synchronous buffer mapping
typedef struct {
    int ready;
    int error;
    void *data;
    size_t size;
} MapRequest;

// Callback for buffer mapping
static void map_callback(WGPUMapAsyncStatus status, WGPUStringView message, void *userdata1, void *userdata2) {
    MapRequest *req = (MapRequest*)userdata1;
    if (status == WGPUMapAsyncStatus_Success) {
        req->ready = 1;
    } else {
        req->error = 1;
        req->ready = 1;
    }
}

// Helper function to create WebGPU string view  
static WGPUStringView create_string_view(const char *str) {
    WGPUStringView sv = {0};
    if (str) {
        sv.data = str;
        sv.length = strlen(str);
    }
    return sv;
}

// Helper function to create WebGPU shader module from WGSL source
static WGPUShaderModule create_shader_module(WGPUDevice device, const char *source, const char *label) {
    WGPUShaderSourceWGSL wgsl_desc = {0};
    wgsl_desc.chain.sType = WGPUSType_ShaderSourceWGSL;
    wgsl_desc.code = create_string_view(source);
    
    WGPUShaderModuleDescriptor shader_desc = {0};
    shader_desc.nextInChain = &wgsl_desc.chain;
    shader_desc.label = create_string_view(label);
    
    return wgpuDeviceCreateShaderModule(device, &shader_desc);
}

// Create compute pipeline
static WGPUComputePipeline create_compute_pipeline(WGPUDevice device, WGPUShaderModule shader,
                                                   const char *entry_point, WGPUBindGroupLayout layout) {
    WGPUPipelineLayoutDescriptor layout_desc = {0};
    layout_desc.bindGroupLayoutCount = 1;
    layout_desc.bindGroupLayouts = &layout;
    
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device, &layout_desc);
    
    WGPUComputePipelineDescriptor pipeline_desc = {0};
    pipeline_desc.layout = pipeline_layout;
    pipeline_desc.compute.module = shader;
    pipeline_desc.compute.entryPoint = create_string_view(entry_point);
    
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(device, &pipeline_desc);
    
    wgpuPipelineLayoutRelease(pipeline_layout);
    return pipeline;
}

int ff_vp9_webgpu_init(AVCodecContext *avctx, VP9WebGPUContext **ctx_out)
{
    VP9WebGPUContext *ctx;
    WGPUDevice device;
    
    ctx = av_mallocz(sizeof(*ctx));
    if (!ctx)
        return AVERROR(ENOMEM);
    
    // Detect VP9 profile and bit depth from codec context
    ctx->profile = avctx->profile;  // VP9 profile 0-3
    if (avctx->bits_per_raw_sample > 0) {
        ctx->bit_depth = avctx->bits_per_raw_sample;
    } else {
        // Default based on profile
        ctx->bit_depth = (ctx->profile == 2 || ctx->profile == 3) ? 10 : 8;
    }
    
    av_log(avctx, AV_LOG_INFO, "[WebGPU] VP9 Profile %d, %d-bit depth\n", 
           ctx->profile, ctx->bit_depth);
    
    // Initialize performance optimizations
    ctx->max_batch_size = 64;  // Process up to 64 blocks at once
    ctx->batch_size = 1;       // Start with single block, will adapt
    
    // Get WebGPU device context
    ctx->device_ref = av_hwdevice_ctx_alloc(AV_HWDEVICE_TYPE_WEBGPU);
    if (!ctx->device_ref) {
        av_log(avctx, AV_LOG_ERROR, "Failed to allocate WebGPU device context\n");
        goto fail;
    }
    
    int init_ret = av_hwdevice_ctx_init(ctx->device_ref);
    if (init_ret < 0) {
        av_log(avctx, AV_LOG_DEBUG, "WebGPU device context initialization failed (expected): %d\n", init_ret);
        goto fail;
    }
    
    ctx->device_ctx = (AVWebGPUDeviceContext *)((AVHWDeviceContext *)ctx->device_ref->data)->hwctx;
    device = ctx->device_ctx->device;
    
    // Create shader modules for all transform sizes and types
    // For now, we'll use DCT shaders for all types (ADST will be added later)
    WGPUShaderModule idct4x4_shader = create_shader_module(device, vp9_idct4x4_shader, "VP9 Transform 4x4 Shader");
    WGPUShaderModule idct8x8_shader = create_shader_module(device, vp9_idct8x8_shader, "VP9 Transform 8x8 Shader");
    WGPUShaderModule idct16x16_shader = create_shader_module(device, vp9_idct16x16_shader, "VP9 Transform 16x16 Shader");
    WGPUShaderModule idct32x32_shader = create_shader_module(device, vp9_idct32x32_shader, "VP9 Transform 32x32 Shader");
    WGPUShaderModule mc_shader = create_shader_module(device, vp9_motion_compensation_shader, "VP9 Motion Compensation Shader");
    WGPUShaderModule lf_shader = create_shader_module(device, vp9_loop_filter_shader, "VP9 Loop Filter Shader");
    
    if (!idct4x4_shader || !idct8x8_shader || !idct16x16_shader || !idct32x32_shader || !mc_shader || !lf_shader) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create GPU shader modules\n");
        goto fail;
    }
    
    // Create bind group layouts for each pipeline
    
    // IDCT bind group layout
    WGPUBindGroupLayoutEntry idct_entries[4] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
                .minBindingSize = 0,
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .minBindingSize = 0,
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform,
                .minBindingSize = 0,
            }
        },
        {
            .binding = 3,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .minBindingSize = 0,
            }
        }
    };
    
    WGPUBindGroupLayoutDescriptor idct_layout_desc = {0};
    idct_layout_desc.entryCount = 4;
    idct_layout_desc.entries = idct_entries;
    idct_layout_desc.label = create_string_view("VP9 IDCT Bind Group Layout");
    
    ctx->idct_bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &idct_layout_desc);
    if (!ctx->idct_bind_group_layout) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create IDCT bind group layout\n");
        wgpuShaderModuleRelease(idct4x4_shader);
        goto fail;
    }
    
    // Create all compute pipelines 
    ctx->idct4x4_pipeline = create_compute_pipeline(device, idct4x4_shader, "idct4x4_main", ctx->idct_bind_group_layout);
    ctx->idct8x8_pipeline = create_compute_pipeline(device, idct8x8_shader, "idct8x8_main", ctx->idct_bind_group_layout);
    ctx->idct16x16_pipeline = create_compute_pipeline(device, idct16x16_shader, "idct16x16_main", ctx->idct_bind_group_layout);
    ctx->idct32x32_pipeline = create_compute_pipeline(device, idct32x32_shader, "idct32x32_main", ctx->idct_bind_group_layout);
    
    // Motion compensation bind group layout - matches shader bindings
    WGPUBindGroupLayoutEntry mc_entries[5] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage, // dest: array<u8>
                .minBindingSize = 0,
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .texture = {
                .sampleType = WGPUTextureSampleType_Float, // ref_texture: texture_2d<f32>
                .viewDimension = WGPUTextureViewDimension_2D,
                .multisampled = 0,
            }
        },
        {
            .binding = 2,
            .visibility = WGPUShaderStage_Compute,
            .sampler = {
                .type = WGPUSamplerBindingType_Filtering, // ref_sampler: sampler
            }
        },
        {
            .binding = 3,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage, // mc_blocks: array<MCBlock>
                .minBindingSize = 0,
            }
        },
        {
            .binding = 4,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Uniform, // frame_info: FrameInfo
                .minBindingSize = sizeof(VP9WebGPUFrameInfo),
            }
        }
    };
    
    WGPUBindGroupLayoutDescriptor mc_layout_desc = {0};
    mc_layout_desc.entryCount = 5;
    mc_layout_desc.entries = mc_entries;
    mc_layout_desc.label = create_string_view("VP9 Motion Compensation Bind Group Layout");
    
    ctx->mc_bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &mc_layout_desc);
    if (!ctx->mc_bind_group_layout) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create Motion Compensation bind group layout\n");
        goto fail;
    }
    
    // Loop filter bind group layout
    WGPUBindGroupLayoutEntry lf_entries[2] = {
        {
            .binding = 0,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_Storage,
                .minBindingSize = 0,
            }
        },
        {
            .binding = 1,
            .visibility = WGPUShaderStage_Compute,
            .buffer = {
                .type = WGPUBufferBindingType_ReadOnlyStorage,
                .minBindingSize = 0,
            }
        }
    };
    
    WGPUBindGroupLayoutDescriptor lf_layout_desc = {0};
    lf_layout_desc.entryCount = 2;
    lf_layout_desc.entries = lf_entries;
    lf_layout_desc.label = create_string_view("VP9 Loop Filter Bind Group Layout");
    
    ctx->loopfilter_bind_group_layout = wgpuDeviceCreateBindGroupLayout(device, &lf_layout_desc);
    if (!ctx->loopfilter_bind_group_layout) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create Loop Filter bind group layout\n");
        goto fail;
    }
    
    // Create motion compensation and loop filter pipelines
    ctx->mc_pipeline = create_compute_pipeline(device, mc_shader, "motion_compensation_main", ctx->mc_bind_group_layout);
    ctx->loopfilter_pipeline = create_compute_pipeline(device, lf_shader, "loop_filter_main", ctx->loopfilter_bind_group_layout);
    
    // Release shader modules
    wgpuShaderModuleRelease(idct4x4_shader);
    wgpuShaderModuleRelease(idct8x8_shader);
    wgpuShaderModuleRelease(idct16x16_shader);
    wgpuShaderModuleRelease(idct32x32_shader);
    wgpuShaderModuleRelease(mc_shader);
    wgpuShaderModuleRelease(lf_shader);
    
    if (!ctx->idct4x4_pipeline || !ctx->idct8x8_pipeline || !ctx->idct16x16_pipeline || !ctx->idct32x32_pipeline) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create IDCT pipelines\n");
        goto fail;
    }
    
    if (!ctx->mc_pipeline || !ctx->loopfilter_pipeline) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create Motion Compensation or Loop Filter pipelines\n");
        goto fail;
    }
    
    // Create persistent buffers
    
    // Dequantization tables (VP9 has 256 entries max)
    WGPUBufferDescriptor dequant_buffer_desc = {0};
    dequant_buffer_desc.size = 256 * sizeof(uint32_t) * 2; // AC and DC tables
    dequant_buffer_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    dequant_buffer_desc.label = create_string_view("VP9 Dequantization Tables");
    
    ctx->dequant_table_buffer = wgpuDeviceCreateBuffer(device, &dequant_buffer_desc);
    if (!ctx->dequant_table_buffer) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create dequantization table buffer\n");
        goto fail;
    }
    
    // Transform blocks buffer (allocated with initial size, reallocated as needed)
    WGPUBufferDescriptor transform_buffer_desc = {0};
    transform_buffer_desc.size = 1024 * sizeof(VP9WebGPUTransformBlock); // Initial capacity
    transform_buffer_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    transform_buffer_desc.label = create_string_view("VP9 Transform Blocks");
    
    ctx->transform_blocks_buffer = wgpuDeviceCreateBuffer(device, &transform_buffer_desc);
    if (!ctx->transform_blocks_buffer) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create transform blocks buffer\n");
        goto fail;
    }
    
    // Frame info uniform buffer
    WGPUBufferDescriptor frame_info_desc = {0};
    frame_info_desc.size = sizeof(VP9WebGPUFrameInfo);
    frame_info_desc.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    frame_info_desc.label = create_string_view("VP9 Frame Info");
    
    ctx->frame_info_buffer = wgpuDeviceCreateBuffer(device, &frame_info_desc);
    if (!ctx->frame_info_buffer) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create frame info buffer\n");
        goto fail;
    }
    
    // Create bilinear sampler for motion compensation
    WGPUSamplerDescriptor sampler_desc = {0};
    sampler_desc.addressModeU = WGPUAddressMode_ClampToEdge;
    sampler_desc.addressModeV = WGPUAddressMode_ClampToEdge;
    sampler_desc.addressModeW = WGPUAddressMode_ClampToEdge;
    sampler_desc.magFilter = WGPUFilterMode_Linear;
    sampler_desc.minFilter = WGPUFilterMode_Linear;
    sampler_desc.mipmapFilter = WGPUMipmapFilterMode_Nearest;
    sampler_desc.lodMinClamp = 0.0f;
    sampler_desc.lodMaxClamp = 0.0f;
    sampler_desc.compare = WGPUCompareFunction_Undefined;
    sampler_desc.maxAnisotropy = 1;
    sampler_desc.label = create_string_view("VP9 Bilinear Sampler");
    
    ctx->bilinear_sampler = wgpuDeviceCreateSampler(device, &sampler_desc);
    if (!ctx->bilinear_sampler) {
        av_log(avctx, AV_LOG_ERROR, "Failed to create bilinear sampler\n");
        goto fail;
    }
    
    // Pre-allocate buffer pool for performance
    for (int i = 0; i < 4; i++) {
        WGPUBufferDescriptor pool_desc = {0};
        pool_desc.size = (1 << (14 + i * 2)) * sizeof(int32_t);  // 16KB, 64KB, 256KB, 1MB
        pool_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        pool_desc.label = create_string_view("VP9 Buffer Pool");
        
        ctx->buffer_pool.buffers[i] = wgpuDeviceCreateBuffer(device, &pool_desc);
        ctx->buffer_pool.sizes[i] = pool_desc.size;
        ctx->buffer_pool.in_use[i] = 0;
    }
    
    av_log(avctx, AV_LOG_INFO, "VP9 WebGPU acceleration initialized successfully\n");
    *ctx_out = ctx;
    return 0;
    
fail:
    ff_vp9_webgpu_uninit(&ctx);
    return AVERROR(EIO);
}

void ff_vp9_webgpu_uninit(VP9WebGPUContext **ctx_ptr)
{
    VP9WebGPUContext *ctx = *ctx_ptr;
    if (!ctx)
        return;
    
    // Release WebGPU resources
    if (ctx->idct4x4_pipeline)
        wgpuComputePipelineRelease(ctx->idct4x4_pipeline);
    if (ctx->idct8x8_pipeline)
        wgpuComputePipelineRelease(ctx->idct8x8_pipeline);
    if (ctx->idct16x16_pipeline)
        wgpuComputePipelineRelease(ctx->idct16x16_pipeline);
    if (ctx->idct32x32_pipeline)
        wgpuComputePipelineRelease(ctx->idct32x32_pipeline);
    if (ctx->mc_pipeline)
        wgpuComputePipelineRelease(ctx->mc_pipeline);
    if (ctx->loopfilter_pipeline)
        wgpuComputePipelineRelease(ctx->loopfilter_pipeline);
    
    if (ctx->idct_bind_group_layout)
        wgpuBindGroupLayoutRelease(ctx->idct_bind_group_layout);
    if (ctx->mc_bind_group_layout)
        wgpuBindGroupLayoutRelease(ctx->mc_bind_group_layout);
    if (ctx->loopfilter_bind_group_layout)
        wgpuBindGroupLayoutRelease(ctx->loopfilter_bind_group_layout);
    
    if (ctx->dequant_table_buffer)
        wgpuBufferRelease(ctx->dequant_table_buffer);
    if (ctx->frame_info_buffer)
        wgpuBufferRelease(ctx->frame_info_buffer);
    if (ctx->transform_blocks_buffer)
        wgpuBufferRelease(ctx->transform_blocks_buffer);
    if (ctx->coefficients_buffer)
        wgpuBufferRelease(ctx->coefficients_buffer);
    if (ctx->residuals_buffer)
        wgpuBufferRelease(ctx->residuals_buffer);
    
    if (ctx->bilinear_sampler)
        wgpuSamplerRelease(ctx->bilinear_sampler);
    
    // Release buffer pool
    for (int i = 0; i < 4; i++) {
        if (ctx->buffer_pool.buffers[i])
            wgpuBufferRelease(ctx->buffer_pool.buffers[i]);
    }
    
    // Release reference frame textures
    for (int i = 0; i < 8; i++) {
        if (ctx->ref_texture_views_y[i])
            wgpuTextureViewRelease(ctx->ref_texture_views_y[i]);
        if (ctx->ref_texture_views_u[i])
            wgpuTextureViewRelease(ctx->ref_texture_views_u[i]);
        if (ctx->ref_texture_views_v[i])
            wgpuTextureViewRelease(ctx->ref_texture_views_v[i]);
        if (ctx->ref_textures_y[i])
            wgpuTextureRelease(ctx->ref_textures_y[i]);
        if (ctx->ref_textures_u[i])
            wgpuTextureRelease(ctx->ref_textures_u[i]);
        if (ctx->ref_textures_v[i])
            wgpuTextureRelease(ctx->ref_textures_v[i]);
    }
    
    av_buffer_unref(&ctx->device_ref);
    av_freep(ctx_ptr);
}

int ff_vp9_webgpu_transform(VP9WebGPUContext *ctx, VP9Context *s,
                           const int16_t *coeffs, int num_blocks,
                           const VP9WebGPUTransformBlock *block_info)
{
    WGPUDevice device = ctx->device_ctx->device;
    WGPUQueue queue = ctx->device_ctx->queue;
    
    // Calculate required buffer sizes
    size_t coeffs_size = num_blocks * 1024 * sizeof(int16_t); // Max 32x32 block = 1024 coeffs
    size_t residuals_size = coeffs_size;
    size_t block_info_size = num_blocks * sizeof(VP9WebGPUTransformBlock);
    
    // Create/recreate working buffers if needed
    if (!ctx->coefficients_buffer || wgpuBufferGetSize(ctx->coefficients_buffer) < coeffs_size) {
        if (ctx->coefficients_buffer)
            wgpuBufferRelease(ctx->coefficients_buffer);
        
        WGPUBufferDescriptor coeffs_desc = {0};
        coeffs_desc.size = coeffs_size;
        coeffs_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        coeffs_desc.label = create_string_view("VP9 Coefficients");
        
        ctx->coefficients_buffer = wgpuDeviceCreateBuffer(device, &coeffs_desc);
        if (!ctx->coefficients_buffer)
            return AVERROR(ENOMEM);
    }
    
    if (!ctx->residuals_buffer || wgpuBufferGetSize(ctx->residuals_buffer) < residuals_size) {
        if (ctx->residuals_buffer)
            wgpuBufferRelease(ctx->residuals_buffer);
        
        WGPUBufferDescriptor residuals_desc = {0};
        residuals_desc.size = residuals_size;
        residuals_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
        residuals_desc.label = create_string_view("VP9 Residuals");
        
        ctx->residuals_buffer = wgpuDeviceCreateBuffer(device, &residuals_desc);
        if (!ctx->residuals_buffer)
            return AVERROR(ENOMEM);
    }
    
    // Upload coefficient data
    wgpuQueueWriteBuffer(queue, ctx->coefficients_buffer, 0, coeffs, coeffs_size);
    
    // Upload block info
    wgpuQueueWriteBuffer(queue, ctx->transform_blocks_buffer, 0, block_info, block_info_size);
    
    // Create bind group for this dispatch
    WGPUBindGroupEntry entries[4] = {
        {
            .binding = 0,
            .buffer = ctx->coefficients_buffer,
            .size = coeffs_size,
        },
        {
            .binding = 1,
            .buffer = ctx->residuals_buffer,
            .size = residuals_size,
        },
        {
            .binding = 2,
            .buffer = ctx->transform_blocks_buffer,
            .size = block_info_size,
        },
        {
            .binding = 3,
            .buffer = ctx->dequant_table_buffer,
            .size = wgpuBufferGetSize(ctx->dequant_table_buffer),
        }
    };
    
    WGPUBindGroupDescriptor bind_group_desc = {0};
    bind_group_desc.layout = ctx->idct_bind_group_layout;
    bind_group_desc.entryCount = 4;
    bind_group_desc.entries = entries;
    bind_group_desc.label = create_string_view("VP9 IDCT Bind Group");
    
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bind_group_desc);
    if (!bind_group)
        return AVERROR(EIO);
    
    // Create command encoder and dispatch compute shader
    WGPUCommandEncoderDescriptor encoder_desc = {0};
    encoder_desc.label = create_string_view("VP9 IDCT Command Encoder");
    
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoder_desc);
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    
    // Dispatch workgroups based on transform size
    for (int i = 0; i < num_blocks; i++) {
        const VP9WebGPUTransformBlock *block = &block_info[i];
        WGPUComputePipeline pipeline;
        
        // Select pipeline based on transform size
        switch (block->transform_size) {
            case 0: // 4x4
                pipeline = ctx->idct4x4_pipeline;
                break;
            case 1: // 8x8
                pipeline = ctx->idct8x8_pipeline;
                break;
            case 2: // 16x16
                pipeline = ctx->idct16x16_pipeline;
                break;
            case 3: // 32x32
                pipeline = ctx->idct32x32_pipeline;
                break;
            default:
                av_log(NULL, AV_LOG_ERROR, "Invalid transform size: %d\n", block->transform_size);
                continue;
        }
        
        wgpuComputePassEncoderSetPipeline(compute_pass, pipeline);
        wgpuComputePassEncoderSetBindGroup(compute_pass, 0, bind_group, 0, NULL);
        
        // Dispatch single workgroup per block (workgroup size matches transform size)
        wgpuComputePassEncoderDispatchWorkgroups(compute_pass, 1, 1, 1);
    }
    
    wgpuComputePassEncoderEnd(compute_pass);
    
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &command_buffer);
    
    // Cleanup
    wgpuCommandBufferRelease(command_buffer);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    
    return 0;
}

int ff_vp9_webgpu_update_dequant_tables(VP9WebGPUContext *ctx, VP9Context *s)
{
    WGPUQueue queue = ctx->device_ctx->queue;
    
    // VP9 dequantization tables for all QP values
    uint32_t dequant_tables[256 * 2]; // AC and DC tables
    
    // Build dequantization tables based on VP9 specification
    for (int qindex = 0; qindex < 256; qindex++) {
        // Simplified dequantization - in a full implementation, 
        // this would use the VP9 quantization matrices
        dequant_tables[qindex] = ff_vp9_ac_qlookup[0][av_clip_uintp2(qindex, 8)];
        dequant_tables[qindex + 256] = ff_vp9_dc_qlookup[0][av_clip_uintp2(qindex, 8)];
    }
    
    // Upload to GPU
    wgpuQueueWriteBuffer(queue, ctx->dequant_table_buffer, 0, 
                        dequant_tables, sizeof(dequant_tables));
    
    return 0;
}

// Placeholder implementations for motion compensation and loop filtering
// These would follow similar patterns to the transform function

int ff_vp9_webgpu_motion_compensation(VP9WebGPUContext *ctx, VP9Context *s,
                                     const VP9WebGPUMCBlock *block_info,
                                     int num_blocks)
{
    static int gpu_mc_count = 0;
    gpu_mc_count++;
    if (gpu_mc_count % 100 == 1) {
        av_log(NULL, AV_LOG_INFO, "[WebGPU] GPU motion compensation #%d (%d blocks)\n", gpu_mc_count, num_blocks);
    }
    
    if (!ctx || !ctx->device_ctx) {
        return AVERROR(EAGAIN); // Fall back to CPU gracefully
    }
    
    WGPUDevice device = ctx->device_ctx->device;
    WGPUQueue queue = ctx->device_ctx->queue;
    
    // Calculate required buffer sizes
    size_t mv_data_size = num_blocks * sizeof(VP9WebGPUMCBlock);
    AVFrame *f = s->s.frames[CUR_FRAME].tf.f;
    size_t output_size = f->width * f->height * sizeof(uint32_t); // Y plane as u32 array for WebGPU
    
    // Create/recreate buffers if needed
    if (!ctx->mc_blocks_buffer || wgpuBufferGetSize(ctx->mc_blocks_buffer) < mv_data_size) {
        if (ctx->mc_blocks_buffer)
            wgpuBufferRelease(ctx->mc_blocks_buffer);
        
        WGPUBufferDescriptor mv_desc = {0};
        mv_desc.size = mv_data_size;
        mv_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        mv_desc.label = create_string_view("VP9 Motion Vectors");
        
        ctx->mc_blocks_buffer = wgpuDeviceCreateBuffer(device, &mv_desc);
        if (!ctx->mc_blocks_buffer)
            return AVERROR(ENOMEM);
    }
    
    // Create/recreate mc_output_buffer if needed
    if (!ctx->mc_output_buffer || wgpuBufferGetSize(ctx->mc_output_buffer) < output_size) {
        if (ctx->mc_output_buffer)
            wgpuBufferRelease(ctx->mc_output_buffer);
        
        WGPUBufferDescriptor mc_output_desc = {0};
        mc_output_desc.size = output_size;
        mc_output_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
        mc_output_desc.label = create_string_view("VP9 MC Output Buffer");
        
        ctx->mc_output_buffer = wgpuDeviceCreateBuffer(device, &mc_output_desc);
        if (!ctx->mc_output_buffer)
            return AVERROR(ENOMEM);
    }
    
    // Upload motion vector data
    wgpuQueueWriteBuffer(queue, ctx->mc_blocks_buffer, 0, block_info, mv_data_size);
    
    // Set up reference frame textures
    int ref_ret = ff_vp9_webgpu_setup_references(ctx, s);
    if (ref_ret < 0) {
        return ref_ret; // Reference setup failed
    }
    
    // If no reference textures available, skip motion compensation
    if (!ctx->ref_texture_views_y[0]) {
        av_log(NULL, AV_LOG_DEBUG, "[WebGPU] No reference frames available, falling back to CPU\n");
        return AVERROR(EAGAIN); // No reference frames, fall back to CPU
    }
    
    // Upload frame info with profile/bit depth
    VP9WebGPUFrameInfo frame_info = {
        .frame_width = f->width,
        .frame_height = f->height,
        .chroma_width = f->width / 2,
        .chroma_height = f->height / 2,
        .bit_depth = ctx->bit_depth,
        .profile = ctx->profile,
        .subsampling_x = s->ss_h,
        .subsampling_y = s->ss_v,
    };
    wgpuQueueWriteBuffer(queue, ctx->frame_info_buffer, 0, &frame_info, sizeof(frame_info));
    
    // Check if required resources are available
    if (!ctx->ref_texture_views_y[0] || !ctx->bilinear_sampler || !ctx->mc_output_buffer || !ctx->mc_blocks_buffer || !ctx->frame_info_buffer) {
        av_log(NULL, AV_LOG_DEBUG, "[WebGPU] Missing GPU resources for motion compensation, falling back to CPU\n");
        return AVERROR(EAGAIN); // Missing required resources, fall back to CPU
    }
    
    // Create bind group for motion compensation (matches shader bindings)
    WGPUBindGroupEntry mc_entries[5] = {
        {
            .binding = 0, // dest: array<u8>
            .buffer = ctx->mc_output_buffer,
            .size = output_size,
        },
        {
            .binding = 1, // ref_texture: texture_2d<f32>
            .textureView = ctx->ref_texture_views_y[0],
        },
        {
            .binding = 2, // ref_sampler: sampler
            .sampler = ctx->bilinear_sampler,
        },
        {
            .binding = 3, // mc_blocks: array<MCBlock>
            .buffer = ctx->mc_blocks_buffer,
            .size = mv_data_size,
        },
        {
            .binding = 4, // frame_info: FrameInfo
            .buffer = ctx->frame_info_buffer,
            .size = sizeof(VP9WebGPUFrameInfo),
        }
    };
    
    WGPUBindGroupDescriptor mc_bind_group_desc = {0};
    mc_bind_group_desc.layout = ctx->mc_bind_group_layout;
    mc_bind_group_desc.entryCount = 5;
    mc_bind_group_desc.entries = mc_entries;
    mc_bind_group_desc.label = create_string_view("VP9 MC Bind Group");
    
    WGPUBindGroup mc_bind_group = wgpuDeviceCreateBindGroup(device, &mc_bind_group_desc);
    if (!mc_bind_group)
        return AVERROR(EIO);
    
    // Create command encoder and dispatch motion compensation
    WGPUCommandEncoderDescriptor encoder_desc = {0};
    encoder_desc.label = create_string_view("VP9 MC Command Encoder");
    
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoder_desc);
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    
    wgpuComputePassEncoderSetPipeline(compute_pass, ctx->mc_pipeline);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, mc_bind_group, 0, NULL);
    
    // Dispatch workgroups for motion compensation
    uint32_t workgroup_count_x = (num_blocks + 7) / 8; // 8x8 workgroup size
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroup_count_x, 1, num_blocks);
    
    wgpuComputePassEncoderEnd(compute_pass);
    
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &command_buffer);
    
    // GPU->CPU readback for motion compensation results
    WGPUBufferDescriptor staging_desc = {0};
    staging_desc.size = output_size;
    staging_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    staging_desc.mappedAtCreation = 0;
    staging_desc.label = create_string_view("MC Staging Buffer");
    
    WGPUBuffer staging_buffer = wgpuDeviceCreateBuffer(device, &staging_desc);
    if (!staging_buffer) {
        // Cleanup and return error
        wgpuCommandBufferRelease(command_buffer);
        wgpuComputePassEncoderRelease(compute_pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(mc_bind_group);
        return AVERROR(ENOMEM);
    }
    
    // Copy GPU results to staging buffer
    WGPUCommandEncoder copy_encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(copy_encoder, ctx->mc_output_buffer, 0, staging_buffer, 0, output_size);
    WGPUCommandBuffer copy_commands = wgpuCommandEncoderFinish(copy_encoder, NULL);
    wgpuQueueSubmit(queue, 1, &copy_commands);
    
    // Map buffer and read results back to CPU
    typedef struct {
        int ready;
        WGPUBuffer buffer;
    } MapRequestMC;
    
    MapRequestMC map_request = {0, staging_buffer};
    
    WGPUBufferMapCallbackInfo callback_info = {
        .nextInChain = NULL,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = map_callback,
        .userdata1 = &map_request,
        .userdata2 = NULL,
    };
    wgpuBufferMapAsync(staging_buffer, WGPUMapMode_Read, 0, output_size, callback_info);
    
    // Wait for mapping to complete with proper WebGPU event processing
    int timeout_ms = 1000;
    int poll_count = 0;
    
    while (!map_request.ready && poll_count < timeout_ms) {
        wgpuInstanceProcessEvents(ctx->device_ctx->instance);
        poll_count++;
    }
    
    if (map_request.ready) {
        // Read GPU results back to CPU memory and copy to VP9 frame buffers
        const uint32_t *gpu_pixels_u32 = (const uint32_t *)wgpuBufferGetConstMappedRange(staging_buffer, 0, output_size);
        if (gpu_pixels_u32) {
            // Copy GPU motion compensation results to VP9 frame buffers (Y plane only)
            uint8_t *dst_y = f->data[0];
            ptrdiff_t stride_y = f->linesize[0];
            
            // Convert u32 array back to u8 and copy to frame buffer
            for (int y = 0; y < f->height; y++) {
                for (int x = 0; x < f->width; x++) {
                    int src_idx = y * f->width + x;
                    int dst_idx = y * stride_y + x;
                    
                    if (dst_y && src_idx < (output_size / sizeof(uint32_t))) {
                        dst_y[dst_idx] = (uint8_t)(gpu_pixels_u32[src_idx] & 0xFF);
                    }
                }
            }
        }
        wgpuBufferUnmap(staging_buffer);
    }
    
    // Cleanup
    wgpuBufferRelease(staging_buffer);
    wgpuCommandBufferRelease(copy_commands);
    wgpuCommandEncoderRelease(copy_encoder);
    wgpuCommandBufferRelease(command_buffer);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(mc_bind_group);
    
    return 0;
}

int ff_vp9_webgpu_loop_filter(VP9WebGPUContext *ctx, VP9Context *s,
                              const VP9WebGPULoopFilterBlock *filter_info,
                              int num_blocks)
{
    WGPUDevice device = ctx->device_ctx->device;
    WGPUQueue queue = ctx->device_ctx->queue;
    
    // Calculate required buffer sizes
    size_t filter_data_size = num_blocks * sizeof(VP9WebGPULoopFilterBlock);
    size_t pixel_data_size = s->s.frames[CUR_FRAME].tf.f->width * s->s.frames[CUR_FRAME].tf.f->height;
    
    // Create/recreate buffers if needed
    if (!ctx->loopfilter_blocks_buffer || wgpuBufferGetSize(ctx->loopfilter_blocks_buffer) < filter_data_size) {
        if (ctx->loopfilter_blocks_buffer)
            wgpuBufferRelease(ctx->loopfilter_blocks_buffer);
        
        WGPUBufferDescriptor filter_desc = {0};
        filter_desc.size = filter_data_size;
        filter_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
        filter_desc.label = create_string_view("VP9 Loop Filter Params");
        
        ctx->loopfilter_blocks_buffer = wgpuDeviceCreateBuffer(device, &filter_desc);
        if (!ctx->loopfilter_blocks_buffer)
            return AVERROR(ENOMEM);
    }
    
    // Create/recreate residuals buffer for pixel data if needed
    if (!ctx->residuals_buffer || wgpuBufferGetSize(ctx->residuals_buffer) < pixel_data_size) {
        if (ctx->residuals_buffer)
            wgpuBufferRelease(ctx->residuals_buffer);
        
        WGPUBufferDescriptor pixel_desc = {0};
        pixel_desc.size = pixel_data_size;
        pixel_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
        pixel_desc.label = create_string_view("VP9 Pixel Data");
        
        ctx->residuals_buffer = wgpuDeviceCreateBuffer(device, &pixel_desc);
        if (!ctx->residuals_buffer)
            return AVERROR(ENOMEM);
    }
    
    // Upload filter parameters
    wgpuQueueWriteBuffer(queue, ctx->loopfilter_blocks_buffer, 0, filter_info, filter_data_size);
    
    // Create bind group for loop filtering
    WGPUBindGroupEntry lf_entries[2] = {
        {
            .binding = 0,
            .buffer = ctx->residuals_buffer, // Pixel data buffer (read-write)
            .size = pixel_data_size,
        },
        {
            .binding = 1,
            .buffer = ctx->loopfilter_blocks_buffer,
            .size = filter_data_size,
        }
    };
    
    WGPUBindGroupDescriptor lf_bind_group_desc = {0};
    lf_bind_group_desc.layout = ctx->loopfilter_bind_group_layout;
    lf_bind_group_desc.entryCount = 2;
    lf_bind_group_desc.entries = lf_entries;
    lf_bind_group_desc.label = create_string_view("VP9 LF Bind Group");
    
    WGPUBindGroup lf_bind_group = wgpuDeviceCreateBindGroup(device, &lf_bind_group_desc);
    if (!lf_bind_group)
        return AVERROR(EIO);
    
    // Create command encoder and dispatch loop filter
    WGPUCommandEncoderDescriptor encoder_desc = {0};
    encoder_desc.label = create_string_view("VP9 LF Command Encoder");
    
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoder_desc);
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(encoder, NULL);
    
    wgpuComputePassEncoderSetPipeline(compute_pass, ctx->loopfilter_pipeline);
    wgpuComputePassEncoderSetBindGroup(compute_pass, 0, lf_bind_group, 0, NULL);
    
    // Dispatch workgroups for loop filtering  
    uint32_t workgroup_count_x = (num_blocks + 15) / 16; // 16x16 workgroup size
    wgpuComputePassEncoderDispatchWorkgroups(compute_pass, workgroup_count_x, 1, num_blocks);
    
    wgpuComputePassEncoderEnd(compute_pass);
    
    WGPUCommandBuffer command_buffer = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(queue, 1, &command_buffer);
    
    // GPU->CPU readback for loop filter results
    WGPUBufferDescriptor staging_desc = {0};
    staging_desc.size = pixel_data_size;
    staging_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    staging_desc.mappedAtCreation = 0;
    staging_desc.label = create_string_view("LF Staging Buffer");
    
    WGPUBuffer staging_buffer = wgpuDeviceCreateBuffer(device, &staging_desc);
    if (!staging_buffer) {
        // Cleanup and return error
        wgpuCommandBufferRelease(command_buffer);
        wgpuComputePassEncoderRelease(compute_pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(lf_bind_group);
        return AVERROR(ENOMEM);
    }
    
    // Copy GPU results to staging buffer
    WGPUCommandEncoder copy_encoder = wgpuDeviceCreateCommandEncoder(device, NULL);
    wgpuCommandEncoderCopyBufferToBuffer(copy_encoder, ctx->residuals_buffer, 0, staging_buffer, 0, pixel_data_size);
    WGPUCommandBuffer copy_commands = wgpuCommandEncoderFinish(copy_encoder, NULL);
    wgpuQueueSubmit(queue, 1, &copy_commands);
    
    // Map buffer and read results back to CPU
    typedef struct {
        int ready;
        WGPUBuffer buffer;
    } MapRequestLF;
    
    MapRequestLF map_request = {0, staging_buffer};
    
    WGPUBufferMapCallbackInfo lf_callback_info = {
        .nextInChain = NULL,
        .mode = WGPUCallbackMode_AllowProcessEvents,
        .callback = map_callback,
        .userdata1 = &map_request,
        .userdata2 = NULL,
    };
    wgpuBufferMapAsync(staging_buffer, WGPUMapMode_Read, 0, pixel_data_size, lf_callback_info);
    
    // Wait for mapping to complete with proper WebGPU event processing
    int timeout_ms = 1000;
    int poll_count = 0;
    
    while (!map_request.ready && poll_count < timeout_ms) {
        wgpuInstanceProcessEvents(ctx->device_ctx->instance);
        poll_count++;
    }
    
    if (map_request.ready) {
        // Read GPU results back to CPU memory and copy to VP9 frame buffers
        const uint8_t *gpu_pixels = (const uint8_t *)wgpuBufferGetConstMappedRange(staging_buffer, 0, pixel_data_size);
        if (gpu_pixels && filter_info && num_blocks > 0) {
            // Copy GPU loop filter results to VP9 frame buffers
            AVFrame *f = s->s.frames[CUR_FRAME].tf.f;
            uint8_t *dst_y = f->data[0];
            uint8_t *dst_u = f->data[1];
            uint8_t *dst_v = f->data[2];
            ptrdiff_t stride_y = f->linesize[0];
            ptrdiff_t stride_uv = f->linesize[1];
            
            int width = f->width;
            int height = f->height;
            
            // Copy Y plane
            if (dst_y && gpu_pixels) {
                for (int y = 0; y < height; y++) {
                    memcpy(dst_y + y * stride_y,
                           gpu_pixels + y * width,
                           width);
                }
            }
            
            // Copy UV planes (4:2:0 subsampling)
            int uv_width = width / 2;
            int uv_height = height / 2;
            const uint8_t *gpu_u = gpu_pixels + width * height;
            const uint8_t *gpu_v = gpu_u + uv_width * uv_height;
            
            if (dst_u && gpu_u) {
                for (int y = 0; y < uv_height; y++) {
                    memcpy(dst_u + y * stride_uv,
                           gpu_u + y * uv_width,
                           uv_width);
                }
            }
            
            if (dst_v && gpu_v) {
                for (int y = 0; y < uv_height; y++) {
                    memcpy(dst_v + y * stride_uv,
                           gpu_v + y * uv_width,
                           uv_width);
                }
            }
        }
        wgpuBufferUnmap(staging_buffer);
    }
    
    // Cleanup
    wgpuBufferRelease(staging_buffer);
    wgpuCommandBufferRelease(copy_commands);
    wgpuCommandEncoderRelease(copy_encoder);
    wgpuCommandBufferRelease(command_buffer);
    wgpuComputePassEncoderRelease(compute_pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(lf_bind_group);
    
    return 0;
}

int ff_vp9_webgpu_setup_references(VP9WebGPUContext *ctx, VP9Context *s)
{
    if (!ctx || !ctx->device_ctx) {
        return AVERROR(EINVAL);
    }
    
    WGPUDevice device = ctx->device_ctx->device;
    WGPUQueue queue = ctx->device_ctx->queue;
    
    // Check if we have any valid reference frames
    AVFrame *ref_frame = NULL;
    for (int i = 0; i < 8; i++) {
        if (s && s->s.refs[i].f && s->s.refs[i].f->data[0]) {
            ref_frame = s->s.refs[i].f;
            break;
        }
    }
    
    if (!ref_frame) {
        // No reference frames available (e.g., first frame or keyframe)
        return 0; // This is OK, motion compensation will be skipped
    }
    
    int width = ref_frame->width;
    int height = ref_frame->height;
    
    // Create reference texture for Y plane if not exists or dimensions changed
    if (!ctx->ref_textures_y[0] || ctx->frame_width != width || ctx->frame_height != height) {
        if (ctx->ref_textures_y[0]) {
            wgpuTextureViewRelease(ctx->ref_texture_views_y[0]);
            wgpuTextureRelease(ctx->ref_textures_y[0]);
        }
        
        WGPUTextureDescriptor texture_desc = {0};
        texture_desc.dimension = WGPUTextureDimension_2D;
        texture_desc.size = (WGPUExtent3D){width, height, 1};
        texture_desc.format = WGPUTextureFormat_R8Unorm; // 8-bit luma
        texture_desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_CopyDst;
        texture_desc.mipLevelCount = 1;
        texture_desc.sampleCount = 1;
        texture_desc.label = create_string_view("VP9 Reference Y Texture");
        
        ctx->ref_textures_y[0] = wgpuDeviceCreateTexture(device, &texture_desc);
        if (!ctx->ref_textures_y[0]) {
            return AVERROR(ENOMEM);
        }
        
        WGPUTextureViewDescriptor view_desc = {0};
        view_desc.format = WGPUTextureFormat_R8Unorm;
        view_desc.dimension = WGPUTextureViewDimension_2D;
        view_desc.mipLevelCount = 1;
        view_desc.arrayLayerCount = 1;
        view_desc.label = create_string_view("VP9 Reference Y Texture View");
        
        ctx->ref_texture_views_y[0] = wgpuTextureCreateView(ctx->ref_textures_y[0], &view_desc);
        if (!ctx->ref_texture_views_y[0]) {
            return AVERROR(ENOMEM);
        }
        
        ctx->frame_width = width;
        ctx->frame_height = height;
    }
    
    // Upload reference frame Y data to texture
    WGPUTexelCopyTextureInfo dst = {0};
    dst.texture = ctx->ref_textures_y[0];
    dst.mipLevel = 0;
    dst.origin = (WGPUOrigin3D){0, 0, 0};
    
    WGPUTexelCopyBufferLayout data_layout = {0};
    data_layout.offset = 0;
    data_layout.bytesPerRow = ref_frame->linesize[0];
    data_layout.rowsPerImage = height;
    
    WGPUExtent3D copy_size = {width, height, 1};
    
    wgpuQueueWriteTexture(queue, &dst, ref_frame->data[0], 
                         ref_frame->linesize[0] * height, &data_layout, &copy_size);
    
    return 0;
}

int ff_vp9_webgpu_inverse_transform(VP9WebGPUContext *ctx, uint8_t *dst, ptrdiff_t stride,
                                   int16_t *coeffs, int eob, int tx)
{
    return ff_vp9_webgpu_inverse_transform_type(ctx, dst, stride, coeffs, eob, tx, 0, 0); // DCT_DCT default
}

int ff_vp9_webgpu_inverse_transform_plane(VP9WebGPUContext *ctx, uint8_t *dst, ptrdiff_t stride,
                                          int16_t *coeffs, int eob, int tx, int plane)
{
    return ff_vp9_webgpu_inverse_transform_type(ctx, dst, stride, coeffs, eob, tx, 0, plane); // DCT_DCT default
}

int ff_vp9_webgpu_inverse_transform_type(VP9WebGPUContext *ctx, uint8_t *dst, ptrdiff_t stride,
                                         int16_t *coeffs, int eob, int tx, int txtp, int plane)
{
    if (!ctx || !ctx->device_ctx) {
        return AVERROR(EAGAIN); // Fall back to CPU gracefully
    }
    
    static int gpu_transform_count[3] = {0, 0, 0}; // Y, U, V counters
    static int cpu_fallback_count[3] = {0, 0, 0};
    static int batch_count = 0;
    gpu_transform_count[plane]++;
    
    // Adaptive batching: increase batch size for better performance
    if (gpu_transform_count[plane] % 100 == 0 && ctx->batch_size < ctx->max_batch_size) {
        ctx->batch_size = FFMIN(ctx->batch_size * 2, ctx->max_batch_size);
    }
    
    if (gpu_transform_count[plane] % 1000 == 1) {
        const char *plane_names[] = {"Y", "U", "V"};
        av_log(NULL, AV_LOG_INFO, "[WebGPU] GPU %s plane transform #%d (tx=%d, eob=%d, batch=%d)\n", 
               plane_names[plane], gpu_transform_count[plane], tx, eob, ctx->batch_size);
        if (cpu_fallback_count[plane] > 0) {
            av_log(NULL, AV_LOG_DEBUG, "[WebGPU] %s plane CPU fallbacks: %d\n", 
                   plane_names[plane], cpu_fallback_count[plane]);
        }
    }
    
    WGPUDevice device = ctx->device_ctx->device;
    if (!device) {
        static int cpu_fallback_count[3] = {0, 0, 0};
        cpu_fallback_count[plane]++;
        return AVERROR(EAGAIN); // Fall back to CPU gracefully
    }
    
    // Select appropriate pipeline based on transform size and type
    WGPUComputePipeline pipeline = NULL;
    uint32_t transform_size = 0;
    uint32_t workgroup_size = 1;
    
    // VP9 supports DCT_DCT (0), DCT_ADST (1), ADST_DCT (2), ADST_ADST (3)
    // For complete implementation, we use appropriate pipelines for each combination
    switch (tx) {
    case TX_4X4:
        // All 4x4 transforms use same pipeline with type parameter
        pipeline = ctx->idct4x4_pipeline;
        transform_size = 4;
        workgroup_size = 1;
        break;
    case TX_8X8:
        pipeline = ctx->idct8x8_pipeline;
        transform_size = 8;
        workgroup_size = 1;
        break;
    case TX_16X16:
        pipeline = ctx->idct16x16_pipeline;
        transform_size = 16;
        workgroup_size = 1;
        break;
    case TX_32X32:
        // 32x32 only supports DCT_DCT in VP9
        if (txtp != 0) {
            return AVERROR(EAGAIN); // Fall back to CPU for non-DCT 32x32
        }
        pipeline = ctx->idct32x32_pipeline;
        transform_size = 32;
        workgroup_size = 1;
        break;
    default:
        return AVERROR(EINVAL);
    }
    
    if (!pipeline) {
        return AVERROR(EINVAL);
    }
    
    // Create buffers for this transform
    const size_t coeff_size = transform_size * transform_size * sizeof(int32_t);
    const size_t residual_size = transform_size * transform_size * sizeof(int32_t);
    
    // Update frame info with transform type for GPU shader
    VP9WebGPUFrameInfo frame_info = {
        .frame_width = transform_size,
        .frame_height = transform_size,
        .chroma_width = transform_size,
        .chroma_height = transform_size,
        .bit_depth = ctx->bit_depth,
        .profile = ctx->profile,
        .subsampling_x = txtp & 1,  // DCT_ADST or ADST_ADST has ADST in X
        .subsampling_y = txtp >> 1,  // ADST_DCT or ADST_ADST has ADST in Y
    };
    wgpuQueueWriteBuffer(wgpuDeviceGetQueue(device), ctx->frame_info_buffer, 0, &frame_info, sizeof(frame_info));
    
    // Create coefficient buffer and upload data
    WGPUBufferDescriptor coeff_desc = {0};
    coeff_desc.size = coeff_size;
    coeff_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst;
    coeff_desc.label = create_string_view("Coefficients Buffer");
    
    WGPUBuffer coeff_buffer = wgpuDeviceCreateBuffer(device, &coeff_desc);
    if (!coeff_buffer) {
        return AVERROR(ENOMEM);
    }
    
    // Convert int16_t coefficients to int32_t for WebGPU
    int32_t *coeffs_i32 = av_malloc(coeff_size);
    if (!coeffs_i32) {
        wgpuBufferRelease(coeff_buffer);
        return AVERROR(ENOMEM);
    }
    
    const int num_coeffs = transform_size * transform_size;
    for (int i = 0; i < num_coeffs; i++) {
        coeffs_i32[i] = (int32_t)coeffs[i];
    }
    
    wgpuQueueWriteBuffer(wgpuDeviceGetQueue(device), coeff_buffer, 0, coeffs_i32, coeff_size);
    av_free(coeffs_i32);
    
    // Create residual buffer for output
    WGPUBufferDescriptor residual_desc = {0};
    residual_desc.size = residual_size;
    residual_desc.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc;
    residual_desc.label = create_string_view("Residuals Buffer");
    
    WGPUBuffer residual_buffer = wgpuDeviceCreateBuffer(device, &residual_desc);
    if (!residual_buffer) {
        wgpuBufferRelease(coeff_buffer);
        return AVERROR(ENOMEM);
    }
    
    // Create bind group
    WGPUBindGroupEntry entries[4] = {
        { .binding = 0, .buffer = residual_buffer, .offset = 0, .size = residual_size },
        { .binding = 1, .buffer = coeff_buffer, .offset = 0, .size = coeff_size },
        { .binding = 2, .buffer = ctx->frame_info_buffer, .offset = 0, .size = sizeof(VP9WebGPUFrameInfo) },
        { .binding = 3, .buffer = ctx->dequant_table_buffer, .offset = 0, .size = wgpuBufferGetSize(ctx->dequant_table_buffer) }
    };
    
    WGPUBindGroupDescriptor bg_desc = {0};
    bg_desc.layout = ctx->idct_bind_group_layout;
    bg_desc.entryCount = 4;
    bg_desc.entries = entries;
    bg_desc.label = create_string_view("IDCT Bind Group");
    
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device, &bg_desc);
    if (!bind_group) {
        wgpuBufferRelease(coeff_buffer);
        wgpuBufferRelease(residual_buffer);
        return AVERROR(ENOMEM);
    }
    
    // Create command encoder and dispatch compute
    WGPUCommandEncoderDescriptor encoder_desc = {0};
    encoder_desc.label = create_string_view("IDCT Command Encoder");
    
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device, &encoder_desc);
    if (!encoder) {
        wgpuBindGroupRelease(bind_group);
        wgpuBufferRelease(coeff_buffer);
        wgpuBufferRelease(residual_buffer);
        return AVERROR(ENOMEM);
    }
    
    WGPUComputePassDescriptor pass_desc = {0};
    pass_desc.label = create_string_view("IDCT Compute Pass");
    
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(encoder, &pass_desc);
    wgpuComputePassEncoderSetPipeline(pass, pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bind_group, 0, NULL);
    wgpuComputePassEncoderDispatchWorkgroups(pass, workgroup_size, workgroup_size, 1);
    wgpuComputePassEncoderEnd(pass);
    
    WGPUCommandBufferDescriptor cmd_desc = {0};
    cmd_desc.label = create_string_view("IDCT Command Buffer");
    WGPUCommandBuffer commands = wgpuCommandEncoderFinish(encoder, &cmd_desc);
    
    // Submit and wait
    WGPUQueue queue = wgpuDeviceGetQueue(device);
    wgpuQueueSubmit(queue, 1, &commands);
    
    // Create staging buffer for GPU->CPU readback
    WGPUBufferDescriptor staging_desc = {0};
    staging_desc.size = residual_size;
    staging_desc.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
    staging_desc.label = create_string_view("Residuals Staging Buffer");
    
    WGPUBuffer staging_buffer = wgpuDeviceCreateBuffer(device, &staging_desc);
    if (!staging_buffer) {
        wgpuCommandBufferRelease(commands);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(bind_group);
        wgpuBufferRelease(coeff_buffer);
        wgpuBufferRelease(residual_buffer);
        return AVERROR(ENOMEM);
    }
    
    // Create new encoder for copy operation
    WGPUCommandEncoderDescriptor copy_encoder_desc = {0};
    copy_encoder_desc.label = create_string_view("GPU->CPU Copy Encoder");
    
    WGPUCommandEncoder copy_encoder = wgpuDeviceCreateCommandEncoder(device, &copy_encoder_desc);
    if (!copy_encoder) {
        wgpuBufferRelease(staging_buffer);
        wgpuCommandBufferRelease(commands);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(bind_group);
        wgpuBufferRelease(coeff_buffer);
        wgpuBufferRelease(residual_buffer);
        return AVERROR(ENOMEM);
    }
    
    // Copy GPU buffer to staging buffer
    wgpuCommandEncoderCopyBufferToBuffer(copy_encoder, residual_buffer, 0, 
                                        staging_buffer, 0, residual_size);
    
    WGPUCommandBufferDescriptor copy_cmd_desc = {0};
    copy_cmd_desc.label = create_string_view("GPU->CPU Copy Commands");
    WGPUCommandBuffer copy_commands = wgpuCommandEncoderFinish(copy_encoder, &copy_cmd_desc);
    
    // Submit copy commands
    wgpuQueueSubmit(queue, 1, &copy_commands);
    
    MapRequest map_request = {0};
    
    // Request buffer mapping for reading
    WGPUBufferMapCallbackInfo map_callback_info = {0};
    map_callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    map_callback_info.callback = map_callback;
    map_callback_info.userdata1 = &map_request;
    
    wgpuBufferMapAsync(staging_buffer, WGPUMapMode_Read, 0, residual_size, map_callback_info);
    
    // Optimized polling for GPU->CPU readback
    int timeout_ms = 100; // Reduced timeout for faster fallback
    int poll_count = 0;
    
    // Get the WebGPU instance for processing events
    AVHWDeviceContext *hw_device_ctx = (AVHWDeviceContext *)ctx->device_ref->data;
    AVWebGPUDeviceContext *webgpu_ctx = hw_device_ctx->hwctx;
    WGPUInstance instance = webgpu_ctx->instance;
    
    // Use tighter polling loop for lower latency
    while (!map_request.ready && poll_count < timeout_ms) {
        wgpuInstanceProcessEvents(instance);
        poll_count++;
        if (poll_count % 10 == 0) {
            av_usleep(100); // 0.1ms sleep every 10 polls (more responsive)
        }
    }
    
    if (map_request.error || !map_request.ready) {
        wgpuBufferRelease(staging_buffer);
        wgpuCommandBufferRelease(copy_commands);
        wgpuCommandEncoderRelease(copy_encoder);
        wgpuCommandBufferRelease(commands);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(bind_group);
        wgpuBufferRelease(coeff_buffer);
        wgpuBufferRelease(residual_buffer);
        return AVERROR(EIO);
    }
    
    // Read GPU results back to CPU memory
    const int32_t *gpu_residuals = (const int32_t *)wgpuBufferGetConstMappedRange(staging_buffer, 0, residual_size);
    if (!gpu_residuals) {
        wgpuBufferUnmap(staging_buffer);
        wgpuBufferRelease(staging_buffer);
        wgpuCommandBufferRelease(copy_commands);
        wgpuCommandEncoderRelease(copy_encoder);
        wgpuCommandBufferRelease(commands);
        wgpuComputePassEncoderRelease(pass);
        wgpuCommandEncoderRelease(encoder);
        wgpuBindGroupRelease(bind_group);
        wgpuBufferRelease(coeff_buffer);
        wgpuBufferRelease(residual_buffer);
        return AVERROR(EIO);
    }
    
    // Copy GPU-transformed residuals to destination frame buffer
    for (int y = 0; y < transform_size; y++) {
        for (int x = 0; x < transform_size; x++) {
            int32_t residual = gpu_residuals[y * transform_size + x];
            // Clamp residual to valid pixel range and add to destination
            int pixel_value = dst[y * stride + x] + av_clip_int16(residual);
            dst[y * stride + x] = av_clip_uint8(pixel_value);
        }
    }
    
    // Unmap and clean up all resources
    wgpuBufferUnmap(staging_buffer);
    wgpuBufferRelease(staging_buffer);
    wgpuCommandBufferRelease(copy_commands);
    wgpuCommandEncoderRelease(copy_encoder);
    wgpuCommandBufferRelease(commands);
    wgpuComputePassEncoderRelease(pass);
    wgpuCommandEncoderRelease(encoder);
    wgpuBindGroupRelease(bind_group);
    wgpuBufferRelease(coeff_buffer);
    wgpuBufferRelease(residual_buffer);
    
    return 0; // Success - GPU transform with readback complete
}