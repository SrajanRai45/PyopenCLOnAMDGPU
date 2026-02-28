<h1>Bypassing CUDA and ROCm: 14X faster then Numpy with PyOpenCL for unsupported hardware</h1>

<h3>Introduction</h3>

Hey, I'm Srajan, a B.Tech Computer Science (Data Science) student. While diving deep into NumPy for machine learning and data processing tasks, I hit a mental roadblock. I became curious about performance limits—specifically, how far CPU-based computation could actually scale before choking.

When I first looked into shifting these workloads to the GPU, the industry's default answer was glaringly obvious: CUDA. Standard deep learning and data libraries like PyTorch and TensorFlow are heavily optimized for Nvidia's ecosystem. But I run an all-AMD setup. CUDA was physically off the table.

Naturally, I pivoted to AMD's ROCm platform. Unfortunately, I hit another wall. My specific GPU—the RX 6500M—didn't make the cut for official ROCm support. I was sitting on teraflops of parallel compute power that the mainstream libraries flat-out refused to talk to.

That left me with PyOpenCL as an absolute last resort.

However, what started as a forced compromise completely shifted my perspective. OpenCL isn't just a fallback; it is a universal key. Because it is an open standard, it can execute on almost anything. I realized that by building an architecture around OpenCL, we aren't just bypassing vendor lock-in—we are unlocking the ability to run high-performance compute tasks on dozens of older, "deprecated" hardware models that the industry has hastily discarded from the compute race.

Drawing on some prior experience working with GPUs in C++ during game development, I decided to stop relying on unsupported frameworks and just write the compute pipelines myself. That curiosity led me down a rabbit hole—starting from a messy, simple Jupyter notebook and eventually evolving it into a fully optimized FastAPI backend.


<h3>💻 My Setup</h3>

Before looking at the performance numbers, you have to understand the environment. I ran all these experiments on my daily driver—an HP Victus laptop powered by an all-AMD hardware configuration.

For my OS, I use Omarchy, an Arch Linux distribution, customized with the Hyprland tiling window manager and Neovim. Building this stripped-down, lightweight environment ensures that background OS bloat doesn't eat into the RAM or CPU threads, giving me a much cleaner baseline for the performance benchmarks.

🖥 The CPU: AMD Ryzen 5 5600H

6 cores / 12 threads

Base clock ~3.3 GHz (Boost up to ~4.2 GHz)

Zen 3 architecture

This chip handles parallel workloads exceptionally well for a general-purpose processor, making it a solid baseline for CPU-bound NumPy calculations.

🎮 The GPU: AMD Radeon RX 6500M

RDNA 2 architecture

1024 stream processors

~5 TFLOPS theoretical compute performance

4GB GDDR6 VRAM

A teraflop (TFLOP) represents one trillion floating-point operations per second. So, in theory, this GPU can crush massive math arrays. However, there is a catch: VRAM. With only 4GB of memory, the VRAM severely limits how much data can be stored and processed on the GPU at one time. Moving data back and forth between the system memory and the GPU is an expensive operation that must be optimized.
