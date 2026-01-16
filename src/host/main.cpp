#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>

// Project headers
#include "sobel_cpu.hpp"
#include "../kernels/sobel_cuda_naive.hpp"
GpuMetrics gpu_metrics;

// Compare CPU and GPU outputs pixel-by-pixel
bool compare_results(const unsigned char* cpu,
                     const unsigned char* gpu,
                     int size)
{
    for (int i = 0; i < size; i++) {
        if (std::abs(cpu[i] - gpu[i]) > 1) {
            std::cout << "Mismatch at index " << i
                      << " | CPU=" << (int)cpu[i]
                      << " GPU=" << (int)gpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

// Main
int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: ./edge_detect <path_to_grayscale_image>\n";
        std::cerr << "Example: ./edge_detect ../data/input/lena.png\n";
        return -1;
    }

    // Load image in grayscale
    cv::Mat img = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << argv[1] << std::endl;
        return -1;
    }

    int width  = img.cols;
    int height = img.rows;
    int size   = width * height;

    std::cout << "Image loaded: " << width << " x " << height << std::endl;

    // Allocate output buffers
    std::vector<unsigned char> cpu_out(size, 0);
    std::vector<unsigned char> gpu_out(size, 0);

    // CPU Sobel — Warm-up + Averaged Timing
    constexpr int CPU_WARMUP_RUNS   = 1;
    constexpr int CPU_MEASURED_RUNS = 5;

    double cpu_total_time_ms = 0.0;

    for (int run = 0; run < CPU_WARMUP_RUNS + CPU_MEASURED_RUNS; run++) {

        auto start = std::chrono::high_resolution_clock::now();

        sobel_cpu(img.data, cpu_out.data(), width, height);

        auto end = std::chrono::high_resolution_clock::now();

        double elapsed =
            std::chrono::duration<double, std::milli>(end - start).count();

        if (run >= CPU_WARMUP_RUNS)
            cpu_total_time_ms += elapsed;
    }

    double cpu_time_ms = cpu_total_time_ms / CPU_MEASURED_RUNS;
    double cpu_avg_time_per_frame = cpu_time_ms;
    double cpu_fps = 1000.0 / cpu_time_ms;
    double cpu_throughput =
        (double)width * height / (cpu_time_ms / 1000.0);

    std::cout << "\n===== CPU Performance (Averaged) =====\n";
    std::cout << "Warm-up runs     : " << CPU_WARMUP_RUNS << "\n";
    std::cout << "Measured runs    : " << CPU_MEASURED_RUNS << "\n";
    std::cout << "Avg Batch time   : " << cpu_time_ms << " ms\n";
    std::cout << "Avg Frame Time   : " << cpu_avg_time_per_frame << " ms\n";
    std::cout << "FPS              : " << cpu_fps << "\n";
    std::cout << "Throughput       : "
              << cpu_throughput / 1e6 << " MPixels/sec\n";
    std::cout << "=====================================\n\n";

    std::cout << "CPU Sobel completed\n";

    // GPU Sobel 
    sobel_cuda_naive(img.data, gpu_out.data(),
                 width, height, gpu_metrics);
    std::cout << "CUDA Sobel completed\n";

    // Save output images
    std::filesystem::path input_path(argv[1]);
    std::string base_name = input_path.stem().string();

    std::string cpu_filename = base_name + "_cpu_sobel.png";
    std::string gpu_filename = base_name + "_gpu_sobel.png";

    cv::imwrite(cpu_filename,
                cv::Mat(height, width, CV_8UC1, cpu_out.data()));
    cv::imwrite(gpu_filename,
                cv::Mat(height, width, CV_8UC1, gpu_out.data()));

    // Validate results
    if (compare_results(cpu_out.data(), gpu_out.data(), size))
        std::cout << "CPU and GPU outputs MATCH ✔\n";
    else
        std::cout << "CPU and GPU outputs DO NOT MATCH ❌\n";

    //CSV benchmark results
    std::filesystem::create_directories("../benchmarks");

    std::string csv_file = "../benchmarks/cpu_gpu_comparison.csv";

    bool write_header = !std::filesystem::exists(csv_file);

    // Open CSV file ONCE
    std::ofstream csv(csv_file, std::ios::app);

    if (!csv.is_open()) {
        std::cerr << "Failed to open CSV file: " << csv_file << "\n";
        return -1;
    }

    if (write_header) {
        csv << "image,width,height,"
            << "cpu_frame_ms,gpu_frame_ms,"
            << "cpu_fps,gpu_fps,"
            << "speedup\n";
}

    // Compute speedup (per-frame)
    double gpu_avg_time_per_frame = gpu_metrics.avg_frame_time_ms;
    double speedup =
        cpu_avg_time_per_frame / gpu_avg_time_per_frame;

    // Write row
    csv << base_name << ","
        << width << ","
        << height << ","
        << cpu_avg_time_per_frame << ","
        << gpu_metrics.avg_frame_time_ms << ","
        << cpu_fps << ","
        << gpu_metrics.fps << ","
        << speedup << "\n";

    csv.close();
    return 0;
}
