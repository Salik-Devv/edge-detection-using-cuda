# Sobel Edge Detection using CUDA

This project implements **Sobel Edge Detection** on both **CPU** and **CUDA-enabled GPU**, and provides a detailed **performance analysis** including FPS, frame time, throughput, speedup, and roofline modeling using NVIDIA Nsight tools.

The goal is to demonstrate how GPU acceleration drastically improves performance for data-parallel image processing workloads, especially at high resolutions (up to **8K images**).

---

## 📁 Project Structure

```
EDGE_DETECTION_USING_CUDA/
│
├── .venv/                  # Python virtual environment (plotting)
├── .vscode/                # VS Code settings
│
├── benchmarks/
│   ├── Plots/              # Generated performance plots
│   │   ├── fps_comparison.png
│   │   ├── frame_time_comparison.png
│   │   ├── speedup_vs_size.png
│   │   └── throughput_scaling.png
│   │
│   └── results/            # Benchmark outputs & profiling artifacts
│       ├── 512x512.png
│       ├── 1024x1024.png
│       ├── 1920x1080.png
│       ├── 5824x3264--6k.png
│       ├── 7680x4320--8k.png
│       ├── cpu_gpu_comparison.csv
│       ├── profile_nsight.png
│       ├── roofline_analysis.png
│       ├── roofline_report.ncu-rep
│       └── sobel_timeline.nsys-rep
│
├── build/                  # Build directory (CMake)
│   ├── edge_detect         # Executable
│   ├── CMakeFiles/
│   ├── CMakeCache.txt
│   └── Makefile
│
├── data/
│   ├── input/              # Input images
│   │   ├── lena.png
│   │   ├── kid.png
│   │   ├── city.png
│   │   ├── city-view.png
│   │   └── trade-center.png
│   │
│   └── output/             # Output images (CPU & GPU)
│       ├── 512×512/
│       ├── 1024×1024/
│       ├── 1920×1080/
│       ├── 5824×3264--6k/
│       └── 7680×4320--8k/
│
├── src/
│   ├── host/
│   │   ├── main.cpp        # Entry point & benchmarking
│   │   ├── sobel_cpu.cpp   # CPU Sobel implementation
│   │   └── sobel_cpu.hpp
│   │
│   ├── kernels/
│   │   ├── sobel_cuda_naive.hpp
│   │   └── sobel_shared.cu # Optimized CUDA kernel
│   │
│   └── utils/
│       ├── plot_results.py # Python plotting script
│       ├── cpu_gpu_comparison.csv
│       ├── fps_comparison.png
│       ├── frame_time_comparison.png
│       ├── speedup_vs_size.png
│       └── throughput_scaling.png
│
├── CMakeLists.txt
├── report.pdf              # Final performance report
└── README.md
```

````

---


## ⚙️ Requirements

### Hardware
- NVIDIA GPU with CUDA support

### Software
- **Ubuntu / WSL2 (recommended)**
- CUDA Toolkit (>= 11.x)
- CMake (>= 3.10)
- OpenCV (for image I/O)
- NVIDIA Nsight Systems & Nsight Compute (for profiling)

Check CUDA installation:
```bash
nvcc --version
````

---

## 🛠️ Build Instructions

From the project root:

```bash
mkdir build
cd build
cmake ..
make -j
```

This will generate the executable:

```bash
./edge_detect
```

---

## ▶️ How to Execute

### Run Sobel on an image

```bash
./edge_detect ../data/input/lena.png
```

You can replace `lena.png` with any image:

```bash
./edge_detect ../data/input/kid.png
./edge_detect ../data/input/city.png
./edge_detect ../data/input/city-view.png
./edge_detect ../data/input/trade-center.png
```

---

## 📊 Runtime Output Explained

The program reports **averaged metrics** for both CPU and GPU:

- Warm-up runs
- Measured runs
- Average frame time (ms)
- FPS
- Throughput (MPixels/sec)
- Output correctness check

Example:

```
CPU Frame Time : 96.21 ms
GPU Frame Time : 0.502 ms
Speedup        : ~191x
CPU and GPU outputs MATCH ✓
```

---

## 🔬 Profiling & Analysis

### Nsight Systems (Timeline)

```bash
nsys profile ./edge_detect ../data/input/city-view.png
```

### Nsight Compute (Kernel Metrics & Roofline)

```bash
ncu ./edge_detect ../data/input/city-view.png
```

Key metrics analyzed:

- Achieved Occupancy (\~91%)
- Kernel Runtime (\~1.65 ms)
- Memory Throughput (\~31.4 GB/s)
- Roofline utilization (\~63%)

---

## 📈 Results Summary

- GPU achieves **up to 246× speedup** over CPU
- GPU frame time remains under **9 ms even for 8K images**
- CPU throughput saturates (\~20 MPixels/sec)
- GPU shows near-optimal occupancy and balanced compute/memory behavior

Full analysis available in the report:
📄 `Sobel_CPU_vs_GPU_CUDA_Report_Enhanced.pdf`

---

## 🎓 Educational Value

This project demonstrates:

- CUDA kernel design
- Memory vs compute trade-offs
- Performance scaling
- Roofline modeling
- Professional GPU profiling methodology


---

## 🚀 Future Improvements

- Shared memory tiling
- Constant memory for Sobel masks
- Kernel fusion
- Multi-stream execution
- FP16 / Tensor Core exploration

---

## 👤 Author

**Mohammad Salik Dev**\
CUDA & GPU Computing Enthusiast

---

If you have questions or want to extend this project, feel free to explore and experiment!

