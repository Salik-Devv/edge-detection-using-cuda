#include "sobel_cuda_naive.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstring>

#define BLOCK_DIM 16
#define NUM_STREAMS 2
#define NUM_FRAMES  8

#define WARMUP_RUNS 3
#define MEASURED_RUNS 10

// Constant memory Sobel kernels
__constant__ int d_sobel_x[9];
__constant__ int d_sobel_y[9];

// Shared-memory Sobel kernel
__global__
void sobel_shared_kernel(const unsigned char* input,
                         unsigned char* output,
                         int width,
                         int height)
{
    __shared__ unsigned char tile[BLOCK_DIM + 2][BLOCK_DIM + 2];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int gx = blockIdx.x * BLOCK_DIM + tx;
    int gy = blockIdx.y * BLOCK_DIM + ty;

    int sx = tx + 1;
    int sy = ty + 1;

    tile[sy][sx] =
        (gx < width && gy < height) ? input[gy * width + gx] : 0;

    if (tx == 0)
        tile[sy][0] = (gx > 0) ? input[gy * width + gx - 1] : 0;
    if (tx == BLOCK_DIM - 1)
        tile[sy][BLOCK_DIM + 1] =
            (gx < width - 1) ? input[gy * width + gx + 1] : 0;
    if (ty == 0)
        tile[0][sx] = (gy > 0) ? input[(gy - 1) * width + gx] : 0;
    if (ty == BLOCK_DIM - 1)
        tile[BLOCK_DIM + 1][sx] =
            (gy < height - 1) ? input[(gy + 1) * width + gx] : 0;

    if (tx == 0 && ty == 0)
        tile[0][0] =
            (gx > 0 && gy > 0)
            ? input[(gy - 1) * width + (gx - 1)] : 0;

    if (tx == BLOCK_DIM - 1 && ty == 0)
        tile[0][BLOCK_DIM + 1] =
            (gx < width - 1 && gy > 0)
            ? input[(gy - 1) * width + (gx + 1)] : 0;

    if (tx == 0 && ty == BLOCK_DIM - 1)
        tile[BLOCK_DIM + 1][0] =
            (gx > 0 && gy < height - 1)
            ? input[(gy + 1) * width + (gx - 1)] : 0;

    if (tx == BLOCK_DIM - 1 && ty == BLOCK_DIM - 1)
        tile[BLOCK_DIM + 1][BLOCK_DIM + 1] =
            (gx < width - 1 && gy < height - 1)
            ? input[(gy + 1) * width + (gx + 1)] : 0;

    __syncthreads();

    if (gx > 0 && gy > 0 && gx < width - 1 && gy < height - 1) {
        int sumX = 0, sumY = 0;
        int idx = 0;

        #pragma unroll
        for (int ky = -1; ky <= 1; ky++) {
            #pragma unroll
            for (int kx = -1; kx <= 1; kx++) {
                int pixel = tile[sy + ky][sx + kx];
                sumX += pixel * d_sobel_x[idx];
                sumY += pixel * d_sobel_y[idx];
                idx++;
            }
        }

        int mag = abs(sumX) + abs(sumY);
        output[gy * width + gx] = (mag > 255) ? 255 : mag;
    }
}

// HOST launcher — Averaged GPU benchmarking 
void sobel_cuda_naive(const unsigned char* h_input,
                      unsigned char* h_output,
                      int width,
                      int height,
                      GpuMetrics& metrics)
{
    size_t size = width * height;

    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++)
        cudaStreamCreate(&streams[i]);

    unsigned char* d_input[NUM_STREAMS];
    unsigned char* d_output[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMalloc(&d_input[i], size);
        cudaMalloc(&d_output[i], size);
    }

    unsigned char* h_in_pinned[NUM_STREAMS];
    unsigned char* h_out_pinned[NUM_STREAMS];

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaMallocHost(&h_in_pinned[i], size);
        cudaMallocHost(&h_out_pinned[i], size);
        memcpy(h_in_pinned[i], h_input, size);
    }

    int h_sobel_x[9] = {-1,0,1,-2,0,2,-1,0,1};
    int h_sobel_y[9] = {-1,-2,-1,0,0,0,1,2,1};

    cudaMemcpyToSymbol(d_sobel_x, h_sobel_x, sizeof(h_sobel_x));
    cudaMemcpyToSymbol(d_sobel_y, h_sobel_y, sizeof(h_sobel_y));

    dim3 block(BLOCK_DIM, BLOCK_DIM);
    dim3 grid((width + BLOCK_DIM - 1) / BLOCK_DIM,
              (height + BLOCK_DIM - 1) / BLOCK_DIM);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float total_time_ms = 0.0f;

    for (int run = 0; run < WARMUP_RUNS + MEASURED_RUNS; run++) {

        cudaEventRecord(start);

        for (int i = 0; i < NUM_FRAMES; i++) {
            int buf = i % NUM_STREAMS;

            cudaMemcpyAsync(d_input[buf], h_in_pinned[buf],
                            size, cudaMemcpyHostToDevice, streams[buf]);

            sobel_shared_kernel<<<grid, block, 0, streams[buf]>>>(
                d_input[buf], d_output[buf], width, height);

            cudaMemcpyAsync(h_out_pinned[buf], d_output[buf],
                            size, cudaMemcpyDeviceToHost, streams[buf]);
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);

        if (run >= WARMUP_RUNS)
            total_time_ms += elapsed_ms;
    }

    float avg_time_ms = total_time_ms / MEASURED_RUNS;
    float avg_time_per_frame = avg_time_ms / NUM_FRAMES;

    memcpy(h_output,
           h_out_pinned[(NUM_FRAMES - 1) % NUM_STREAMS],
           size);

    float fps = (NUM_FRAMES * 1000.0f) / avg_time_ms;
    double throughput =
        (double)NUM_FRAMES * width * height / (avg_time_ms / 1000.0);

    printf("===== GPU Performance (Averaged) =====\n");
    printf("Warm-up runs     : %d\n", WARMUP_RUNS);
    printf("Measured runs    : %d\n", MEASURED_RUNS);
    printf("Avg Batch Time   : %.3f ms\n", avg_time_ms);
    printf("Avg Frame Time   : %.3f ms\n", avg_time_per_frame);
    printf("FPS              : %.2f\n", fps);
    printf("Throughput       : %.2f MPixels/sec\n",
           throughput / 1e6);
    printf("=====================================\n");
    metrics.avg_batch_time_ms = avg_time_ms;
    metrics.avg_frame_time_ms = avg_time_ms / NUM_FRAMES;
    metrics.fps               = fps;
    metrics.throughput_mpix   = throughput / 1e6;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaFree(d_input[i]);
        cudaFree(d_output[i]);
        cudaFreeHost(h_in_pinned[i]);
        cudaFreeHost(h_out_pinned[i]);
        cudaStreamDestroy(streams[i]);
    }
}
