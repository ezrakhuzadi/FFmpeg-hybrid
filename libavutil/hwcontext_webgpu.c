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

#include <webgpu.h>

#include "config.h"
#include "common.h"
#include "hwcontext.h"
#include "hwcontext_internal.h"
#include "hwcontext_webgpu.h"
#include "mem.h"
#include "pixdesc.h"
#include "pixfmt.h"
#include "time.h"

typedef struct WebGPUFramesContext {
    AVWebGPUFramesContext p;

    /**
     * Pool of AVWebGPUFrame objects for reuse.
     */
    AVBufferPool *pool;
} WebGPUFramesContext;

static const struct {
    enum AVPixelFormat pixfmt;
    WGPUTextureFormat webgpu_fmt;
} supported_formats[] = {
    { AV_PIX_FMT_YUV420P,   WGPUTextureFormat_R8Unorm },    // Y plane
    { AV_PIX_FMT_YUV420P10, WGPUTextureFormat_R16Uint },    // 10-bit Y plane
    { AV_PIX_FMT_NV12,      WGPUTextureFormat_R8Unorm },    // Y plane (UV is RG8)
    { AV_PIX_FMT_NONE,      WGPUTextureFormat_Undefined }
};

WGPUTextureFormat av_webgpu_format_from_pixfmt(enum AVPixelFormat pixfmt)
{
    for (int i = 0; supported_formats[i].pixfmt != AV_PIX_FMT_NONE; i++) {
        if (supported_formats[i].pixfmt == pixfmt)
            return supported_formats[i].webgpu_fmt;
    }
    return WGPUTextureFormat_Undefined;
}

AVWebGPUFrame *av_webgpu_frame_alloc(void)
{
    return av_mallocz(sizeof(AVWebGPUFrame));
}

static void webgpu_device_free(AVHWDeviceContext *hwdev)
{
    AVWebGPUDeviceContext *ctx = hwdev->hwctx;
    
    if (ctx->queue)
        wgpuQueueRelease(ctx->queue);
    if (ctx->device)
        wgpuDeviceRelease(ctx->device);
    if (ctx->adapter)
        wgpuAdapterRelease(ctx->adapter);
    if (ctx->instance)
        wgpuInstanceRelease(ctx->instance);
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


// Structure for synchronous adapter/device request
typedef struct {
    WGPUAdapter adapter;
    WGPUDevice device;
    int ready;
    int error;
} SyncRequest;

static void on_adapter_request_ended(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void* userdata2)
{
    SyncRequest* req = (SyncRequest*)userdata1;
    if (status == WGPURequestAdapterStatus_Success) {
        req->adapter = adapter;
        req->ready = 1;
    } else {
        req->error = 1;
        req->ready = 1;
    }
}

static void on_device_request_ended(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void* userdata2)
{
    SyncRequest* req = (SyncRequest*)userdata1;
    if (status == WGPURequestDeviceStatus_Success) {
        req->device = device;
        req->ready = 1;
    } else {
        req->error = 1;
        req->ready = 1;
    }
}

static int webgpu_device_create_sync(AVHWDeviceContext *hwdev)
{
    AVWebGPUDeviceContext *ctx = hwdev->hwctx;
    WGPUInstanceDescriptor instance_desc = {0};
    WGPURequestAdapterOptions adapter_opts = {0};
    WGPUDeviceDescriptor device_desc = {0};
    SyncRequest adapter_req = {0};
    SyncRequest device_req = {0};
    int timeout_ms = 5000; // 5 second timeout
    int poll_count = 0;
    
    // Create WebGPU instance
    ctx->instance = wgpuCreateInstance(&instance_desc);
    if (!ctx->instance) {
        av_log(hwdev, AV_LOG_ERROR, "Failed to create WebGPU instance\n");
        return AVERROR(EIO);
    }
    
    av_log(hwdev, AV_LOG_DEBUG, "WebGPU instance created, requesting adapter...\n");
    
    // Request adapter synchronously
    adapter_opts.powerPreference = WGPUPowerPreference_HighPerformance;
    WGPURequestAdapterCallbackInfo adapter_callback = {0};
    adapter_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    adapter_callback.callback = on_adapter_request_ended;
    adapter_callback.userdata1 = &adapter_req;
    wgpuInstanceRequestAdapter(ctx->instance, &adapter_opts, adapter_callback);
    
    // Poll until adapter request completes
    while (!adapter_req.ready && poll_count < timeout_ms) {
        wgpuInstanceProcessEvents(ctx->instance);
        poll_count++;
        if (poll_count % 100 == 0) {
            av_log(hwdev, AV_LOG_DEBUG, "Waiting for WebGPU adapter... (%d/%d ms)\n", poll_count, timeout_ms);
        }
        av_usleep(1000); // 1ms sleep
    }
    
    if (adapter_req.error || !adapter_req.adapter) {
        av_log(hwdev, AV_LOG_ERROR, "Failed to get WebGPU adapter\n");
        return AVERROR(EIO);
    }
    
    av_log(hwdev, AV_LOG_DEBUG, "WebGPU adapter acquired, requesting device...\n");
    
    // Request device synchronously
    device_desc.label = create_string_view("FFmpeg VP9 WebGPU Device");
    poll_count = 0;
    WGPURequestDeviceCallbackInfo device_callback = {0};
    device_callback.mode = WGPUCallbackMode_WaitAnyOnly;
    device_callback.callback = on_device_request_ended;
    device_callback.userdata1 = &device_req;
    wgpuAdapterRequestDevice(adapter_req.adapter, &device_desc, device_callback);
    
    // Poll until device request completes
    while (!device_req.ready && poll_count < timeout_ms) {
        wgpuInstanceProcessEvents(ctx->instance);
        poll_count++;
        if (poll_count % 100 == 0) {
            av_log(hwdev, AV_LOG_DEBUG, "Waiting for WebGPU device... (%d/%d ms)\n", poll_count, timeout_ms);
        }
        av_usleep(1000); // 1ms sleep
    }
    
    if (device_req.error || !device_req.device) {
        av_log(hwdev, AV_LOG_ERROR, "Failed to get WebGPU device\n");
        wgpuAdapterRelease(adapter_req.adapter);
        return AVERROR(EIO);
    }
    
    av_log(hwdev, AV_LOG_INFO, "WebGPU device created successfully!\n");
    
    ctx->device = device_req.device;
    ctx->queue = wgpuDeviceGetQueue(ctx->device);
    
    // Clean up adapter (device holds a reference)
    wgpuAdapterRelease(adapter_req.adapter);
    
    return 0;
}

static int webgpu_device_init(AVHWDeviceContext *hwdev)
{
    AVWebGPUDeviceContext *ctx = hwdev->hwctx;

    if (!ctx->device) {
        // Try to create device synchronously
        int ret = webgpu_device_create_sync(hwdev);
        if (ret < 0) {
            av_log(hwdev, AV_LOG_WARNING, "Failed to create WebGPU device: %d\n", ret);
            return ret;
        }
    }

    // If device was provided by user or created above, get the queue
    if (!ctx->queue && ctx->device) {
        ctx->queue = wgpuDeviceGetQueue(ctx->device);
        if (!ctx->queue) {
            av_log(hwdev, AV_LOG_ERROR, "Failed to get WebGPU queue\n");
            return AVERROR(EIO);
        }
    }

    return 0;
}

static void webgpu_buffer_free(void *opaque, uint8_t *data)
{
    AVWebGPUFrame *frame = (AVWebGPUFrame *)data;

    for (int i = 0; i < frame->nb_planes; i++) {
        if (frame->texture_view[i])
            wgpuTextureViewRelease(frame->texture_view[i]);
        if (frame->texture[i])
            wgpuTextureRelease(frame->texture[i]);
        if (frame->staging_buffer[i])
            wgpuBufferRelease(frame->staging_buffer[i]);
    }

    av_free(frame);
}

static AVBufferRef *webgpu_pool_alloc(void *opaque, size_t size)
{
    AVHWFramesContext *hwframe_ctx = opaque;
    WebGPUFramesContext *ctx = hwframe_ctx->hwctx;
    AVWebGPUDeviceContext *device_ctx = hwframe_ctx->device_ctx->hwctx;
    AVWebGPUFrame *frame;
    const AVPixFmtDescriptor *desc;
    WGPUTextureDescriptor tex_desc = {0};
    WGPUTextureViewDescriptor view_desc = {0};

    frame = av_webgpu_frame_alloc();
    if (!frame)
        return NULL;
        
    // Avoid unused variable warning
    (void)ctx;

    desc = av_pix_fmt_desc_get(hwframe_ctx->sw_format);
    if (!desc) {
        av_free(frame);
        return NULL;
    }

    frame->format = av_webgpu_format_from_pixfmt(hwframe_ctx->sw_format);
    frame->nb_planes = desc->nb_components;

    // Create textures for each plane
    for (int i = 0; i < frame->nb_planes; i++) {
        int plane_w = hwframe_ctx->width;
        int plane_h = hwframe_ctx->height;

        // Adjust dimensions for chroma planes
        if (i > 0 && desc->log2_chroma_w) {
            plane_w >>= desc->log2_chroma_w;
            plane_h >>= desc->log2_chroma_h;
        }

        frame->plane[i].width = plane_w;
        frame->plane[i].height = plane_h;

        // Create texture
        tex_desc.nextInChain = NULL;
        tex_desc.label = create_string_view(NULL);
        tex_desc.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding |
                        WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc;
        tex_desc.dimension = WGPUTextureDimension_2D;
        tex_desc.size.width = plane_w;
        tex_desc.size.height = plane_h;
        tex_desc.size.depthOrArrayLayers = 1;
        tex_desc.format = frame->format;
        tex_desc.mipLevelCount = 1;
        tex_desc.sampleCount = 1;

        frame->texture[i] = wgpuDeviceCreateTexture(device_ctx->device, &tex_desc);
        if (!frame->texture[i]) {
            av_log(hwframe_ctx, AV_LOG_ERROR, "Failed to create WebGPU texture for plane %d\n", i);
            goto fail;
        }

        // Create texture view
        view_desc.nextInChain = NULL;
        view_desc.label = create_string_view(NULL);
        view_desc.format = frame->format;
        view_desc.dimension = WGPUTextureViewDimension_2D;
        view_desc.baseMipLevel = 0;
        view_desc.mipLevelCount = 1;
        view_desc.baseArrayLayer = 0;
        view_desc.arrayLayerCount = 1;
        view_desc.aspect = WGPUTextureAspect_All;

        frame->texture_view[i] = wgpuTextureCreateView(frame->texture[i], &view_desc);
        if (!frame->texture_view[i]) {
            av_log(hwframe_ctx, AV_LOG_ERROR, "Failed to create WebGPU texture view for plane %d\n", i);
            goto fail;
        }
    }

    return av_buffer_create((uint8_t *)frame, sizeof(*frame), webgpu_buffer_free, NULL, 0);

fail:
    webgpu_buffer_free(NULL, (uint8_t *)frame);
    return NULL;
}

static int webgpu_frames_init(AVHWFramesContext *hwframe_ctx)
{
    WebGPUFramesContext *ctx = hwframe_ctx->hwctx;
    AVWebGPUDeviceContext *device_ctx = hwframe_ctx->device_ctx->hwctx;

    if (!device_ctx->device) {
        av_log(hwframe_ctx, AV_LOG_ERROR, "WebGPU device not initialized\n");
        return AVERROR(EINVAL);
    }

    // Validate format
    ctx->p.format = av_webgpu_format_from_pixfmt(hwframe_ctx->sw_format);
    if (ctx->p.format == WGPUTextureFormat_Undefined) {
        av_log(hwframe_ctx, AV_LOG_ERROR, "Unsupported pixel format: %s\n",
               av_get_pix_fmt_name(hwframe_ctx->sw_format));
        return AVERROR(ENOSYS);
    }

    // Set default usage if not specified
    if (!ctx->p.usage) {
        ctx->p.usage = WGPUTextureUsage_TextureBinding | WGPUTextureUsage_StorageBinding |
                      WGPUTextureUsage_CopyDst | WGPUTextureUsage_CopySrc;
    }

    // Create frame pool
    if (!hwframe_ctx->pool) {
        hwframe_ctx->pool = av_buffer_pool_init2(sizeof(AVWebGPUFrame), hwframe_ctx,
                                                webgpu_pool_alloc, NULL);
        if (!hwframe_ctx->pool)
            return AVERROR(ENOMEM);
    }

    return 0;
}

static int webgpu_get_buffer(AVHWFramesContext *hwframe_ctx, AVFrame *frame)
{
    frame->buf[0] = av_buffer_pool_get(hwframe_ctx->pool);
    if (!frame->buf[0])
        return AVERROR(ENOMEM);

    frame->data[0] = frame->buf[0]->data;
    frame->format = AV_PIX_FMT_WEBGPU;
    frame->width = hwframe_ctx->width;
    frame->height = hwframe_ctx->height;

    return 0;
}

static int webgpu_transfer_get_formats(AVHWFramesContext *hwframe_ctx,
                                      enum AVHWFrameTransferDirection dir,
                                      enum AVPixelFormat **formats)
{
    enum AVPixelFormat *fmts;

    fmts = av_malloc_array(2, sizeof(*fmts));
    if (!fmts)
        return AVERROR(ENOMEM);

    fmts[0] = hwframe_ctx->sw_format;
    fmts[1] = AV_PIX_FMT_NONE;

    *formats = fmts;
    return 0;
}

static int webgpu_transfer_data_to(AVHWFramesContext *hwframe_ctx, AVFrame *dst,
                                  const AVFrame *src)
{
    AVWebGPUFrame *webgpu_frame = (AVWebGPUFrame *)dst->data[0];
    AVWebGPUDeviceContext *device_ctx = hwframe_ctx->device_ctx->hwctx;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(src->format);
    WGPUTexelCopyBufferInfo copy_src = {0};
    WGPUTexelCopyTextureInfo copy_dst = {0};
    WGPUExtent3D copy_size = {0};
    WGPUCommandEncoderDescriptor enc_desc = {0};
    WGPUCommandEncoder encoder;
    WGPUCommandBuffer cmd_buffer;

    if (!desc)
        return AVERROR(EINVAL);

    enc_desc.label = create_string_view("WebGPU Transfer");
    encoder = wgpuDeviceCreateCommandEncoder(device_ctx->device, &enc_desc);
    if (!encoder)
        return AVERROR(EIO);

    for (int i = 0; i < webgpu_frame->nb_planes; i++) {
        int plane_w = webgpu_frame->plane[i].width;
        int plane_h = webgpu_frame->plane[i].height;
        int bytes_per_pixel = desc->comp[i].depth > 8 ? 2 : 1;
        size_t buffer_size = plane_w * plane_h * bytes_per_pixel;

        // Create staging buffer if needed
        if (!webgpu_frame->staging_buffer[i]) {
            WGPUBufferDescriptor buf_desc = {0};
            buf_desc.label = create_string_view("Staging Buffer");
            buf_desc.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_MapWrite;
            buf_desc.size = buffer_size;
            buf_desc.mappedAtCreation = 1;

            webgpu_frame->staging_buffer[i] = wgpuDeviceCreateBuffer(device_ctx->device, &buf_desc);
            if (!webgpu_frame->staging_buffer[i]) {
                wgpuCommandEncoderRelease(encoder);
                return AVERROR(EIO);
            }
        }

        // Copy data to staging buffer
        void *mapped_data = wgpuBufferGetMappedRange(webgpu_frame->staging_buffer[i], 0, buffer_size);
        if (mapped_data) {
            uint8_t *dst_data = mapped_data;
            uint8_t *src_data = src->data[i];
            int src_linesize = src->linesize[i];

            for (int y = 0; y < plane_h; y++) {
                memcpy(dst_data + y * plane_w * bytes_per_pixel,
                       src_data + y * src_linesize,
                       plane_w * bytes_per_pixel);
            }
            wgpuBufferUnmap(webgpu_frame->staging_buffer[i]);
        }

        // Copy from buffer to texture
        copy_src.layout.offset = 0;
        copy_src.layout.bytesPerRow = plane_w * bytes_per_pixel;
        copy_src.layout.rowsPerImage = plane_h;
        copy_src.buffer = webgpu_frame->staging_buffer[i];

        copy_dst.texture = webgpu_frame->texture[i];
        copy_dst.mipLevel = 0;
        copy_dst.origin.x = 0;
        copy_dst.origin.y = 0;
        copy_dst.origin.z = 0;
        copy_dst.aspect = WGPUTextureAspect_All;

        copy_size.width = plane_w;
        copy_size.height = plane_h;
        copy_size.depthOrArrayLayers = 1;

        wgpuCommandEncoderCopyBufferToTexture(encoder, &copy_src, &copy_dst, &copy_size);
    }

    cmd_buffer = wgpuCommandEncoderFinish(encoder, NULL);
    wgpuQueueSubmit(device_ctx->queue, 1, &cmd_buffer);

    wgpuCommandBufferRelease(cmd_buffer);
    wgpuCommandEncoderRelease(encoder);

    return 0;
}

static int webgpu_transfer_data_from(AVHWFramesContext *hwframe_ctx, AVFrame *dst,
                                    const AVFrame *src)
{
    // Similar implementation for GPU->CPU transfer
    // For now, return not implemented
    return AVERROR(ENOSYS);
}

const HWContextType ff_hwcontext_type_webgpu = {
    .type                   = AV_HWDEVICE_TYPE_WEBGPU,
    .name                   = "WebGPU",

    .device_hwctx_size      = sizeof(AVWebGPUDeviceContext),
    .frames_hwctx_size      = sizeof(WebGPUFramesContext),

    .device_init            = webgpu_device_init,
    .device_uninit          = webgpu_device_free,

    .frames_init            = webgpu_frames_init,
    .frames_get_buffer      = webgpu_get_buffer,

    .transfer_get_formats   = webgpu_transfer_get_formats,
    .transfer_data_to       = webgpu_transfer_data_to,
    .transfer_data_from     = webgpu_transfer_data_from,

    .pix_fmts = (const enum AVPixelFormat[]) {
        AV_PIX_FMT_WEBGPU,
        AV_PIX_FMT_NONE
    },
};