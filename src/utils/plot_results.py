import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv("cpu_gpu_comparison.csv")

# Compute megapixels for x-axis
df["megapixels"] = (df["width"] * df["height"]) / 1e6

# Sort by image size
df = df.sort_values("megapixels")

# ==============================
# 1. CPU vs GPU Frame Time
# ==============================
plt.figure(figsize=(8, 5))
plt.plot(df["megapixels"], df["cpu_frame_ms"], marker="o", label="CPU")
plt.plot(df["megapixels"], df["gpu_frame_ms"], marker="o", label="GPU")
plt.yscale("log")
plt.xlabel("Image Size (MPixels)")
plt.ylabel("Frame Time (ms, log scale)")
plt.title("CPU vs GPU Sobel Frame Time")
plt.legend()
plt.grid(True, which="both", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("frame_time_comparison.png", dpi=300)
plt.show()

# ==============================
# 2. Speedup vs Image Size
# ==============================
plt.figure(figsize=(8, 5))
plt.plot(df["megapixels"], df["speedup"], marker="o", color="green")
plt.xlabel("Image Size (MPixels)")
plt.ylabel("Speedup (CPU / GPU)")
plt.title("GPU Speedup over CPU (Sobel Edge Detection)")
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("speedup_vs_size.png", dpi=300)
plt.show()

# ==============================
# 3. FPS Comparison
# ==============================
x = np.arange(len(df))
width = 0.35

plt.figure(figsize=(9, 5))
plt.bar(x - width/2, df["cpu_fps"], width, label="CPU")
plt.bar(x + width/2, df["gpu_fps"], width, label="GPU")

plt.xticks(x, df["image"], rotation=30)
plt.ylabel("FPS")
plt.title("CPU vs GPU FPS Comparison")
plt.legend()
plt.grid(axis="y", linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("fps_comparison.png", dpi=300)
plt.show()

# ==============================
# 4. Throughput (MPixels/sec)
# ==============================
cpu_throughput = df["cpu_fps"] * df["megapixels"]
gpu_throughput = df["gpu_fps"] * df["megapixels"]

plt.figure(figsize=(8, 5))
plt.plot(df["megapixels"], cpu_throughput, marker="o", label="CPU")
plt.plot(df["megapixels"], gpu_throughput, marker="o", label="GPU")
plt.xlabel("Image Size (MPixels)")
plt.ylabel("Throughput (MPixels/sec)")
plt.title("Throughput Scaling with Image Size")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
plt.savefig("throughput_scaling.png", dpi=300)
plt.show()
