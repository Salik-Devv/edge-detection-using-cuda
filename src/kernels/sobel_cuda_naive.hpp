#pragma once

struct GpuMetrics {
    double avg_batch_time_ms;
    double avg_frame_time_ms;
    double fps;
    double throughput_mpix;
};

void sobel_cuda_naive(const unsigned char* h_input,
                      unsigned char* h_output,
                      int width,
                      int height,
                      GpuMetrics& metrics);