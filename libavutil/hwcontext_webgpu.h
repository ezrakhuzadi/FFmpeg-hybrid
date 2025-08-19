/*
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

#ifndef AVUTIL_HWCONTEXT_WEBGPU_H
#define AVUTIL_HWCONTEXT_WEBGPU_H

#include <webgpu.h>

#include "pixfmt.h"
#include "frame.h"
#include "hwcontext.h"

typedef struct AVWebGPUFrame AVWebGPUFrame;

/**
 * @file
 * API-specific header for AV_HWDEVICE_TYPE_WEBGPU.
 */

/**
 * Main WebGPU context, allocated as AVHWDeviceContext.hwctx.
 */
typedef struct AVWebGPUDeviceContext {
    /**
     * WebGPU instance. Set by the user or created automatically.
     */
    WGPUInstance instance;

    /**
     * WebGPU adapter. Set by the user or selected automatically.
     */
    WGPUAdapter adapter;

    /**
     * WebGPU device. Set by the user or created automatically.
     */
    WGPUDevice device;

    /**
     * WebGPU queue for command submission. Usually the default queue.
     */
    WGPUQueue queue;

    /**
     * Features required for VP9 decode operations.
     * Set automatically during device init.
     */
    WGPULimits limits;
    
    /**
     * Supported features for this device.
     */
    WGPULimits supported_limits;
} AVWebGPUDeviceContext;

/**
 * WebGPU frames context, allocated as AVHWFramesContext.hwctx.
 */
typedef struct AVWebGPUFramesContext {
    /**
     * Format of the WebGPU textures. Should be compatible with sw_format.
     */
    WGPUTextureFormat format;

    /**
     * WebGPU texture usage flags. Defaults to TEXTURE_BINDING | STORAGE_BINDING | COPY_SRC | COPY_DST.
     */
    WGPUTextureUsage usage;

    /**
     * Internal frame pool data.
     */
    void *pool_internal;
} AVWebGPUFramesContext;

/**
 * WebGPU frame structure.
 */
struct AVWebGPUFrame {
    /**
     * WebGPU textures for Y, U, V planes (for planar formats).
     * For packed formats, only texture[0] is used.
     */
    WGPUTexture texture[AV_NUM_DATA_POINTERS];

    /**
     * Texture views for compute shader access.
     */
    WGPUTextureView texture_view[AV_NUM_DATA_POINTERS];

    /**
     * Buffer objects for staging data transfer.
     */
    WGPUBuffer staging_buffer[AV_NUM_DATA_POINTERS];

    /**
     * Format of the textures.
     */
    WGPUTextureFormat format;

    /**
     * Dimensions of each texture/plane.
     */
    struct {
        int width, height;
    } plane[AV_NUM_DATA_POINTERS];

    /**
     * Number of planes used.
     */
    int nb_planes;

    /**
     * Internal synchronization data.
     */
    void *sync;
};

/**
 * Allocates a single AVWebGPUFrame and initializes everything as 0.
 * @note Must be freed via av_free()
 */
AVWebGPUFrame *av_webgpu_frame_alloc(void);

/**
 * Get optimal WebGPU texture format for a given pixel format.
 */
WGPUTextureFormat av_webgpu_format_from_pixfmt(enum AVPixelFormat pixfmt);

#endif /* AVUTIL_HWCONTEXT_WEBGPU_H */