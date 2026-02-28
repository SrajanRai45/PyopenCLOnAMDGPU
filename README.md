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


<h3>🏗️ The Problem with Naive GPU Compute</h3>
When I first wrote my raw PyOpenCL script, the results were slower then expected. For small arrays, the "super-powerful" GPU was actually slower than standard NumPy.

To understand why, you have to look at the hardware bridge between the CPU and the GPU: the PCIe bus.

Think about my background in game development. In rendering, we have something called a Draw Call. If you tell the GPU to draw 1,000 objects one by one, the CPU has to individually package and send 1,000 separate instructions. The overhead of crossing that bridge destroys your frame rate. The GPU spends 90% of its time idling, waiting for the CPU to hand it the next tiny piece of work. To fix it, you batch the data.

![Alt text for the image](assets/diagram1.png)

The exact same concept applies to OpenCL compute. A naive script constantly initializes the GPU context, compiles the C-kernel from scratch, pushes a tiny array over the PCIe bus, waits for the compute, and pulls it back. The physical latency of moving the data and rebuilding the environment takes longer than the actual floating-point math. NumPy wins on small datasets because the data never has to leave the system RAM. To beat NumPy, I didn't just need to use the GPU—I needed to eliminate the setup overhead and keep the GPU fed.

### ⏳ The Interruption (and the Pivot)
Right in the middle of wrestling with these memory bottlenecks, university exams hit. I had to shelve the OpenCL experiments to focus on a massive 5-day intensive study plan, specifically grinding through Theory of Computation and Machine Learning.

Ironically, stepping away from the code was exactly what the project needed.

During my study breaks, I started playing around with FastAPI to get a better grip on backend architecture. As I got comfortable with FastAPI’s asynchronous routing and dependency injection, a lightbulb went off.

My OpenCL script was failing because it was structured like a one-off procedural script. It was recompiling the math kernels and rebuilding the GPU command queue every single time I hit "run."

What if I built an API where the GPU context was initialized exactly once? If I could tie the OpenCL environment to the lifecycle of a web server, the endpoints would simply act as funnels, feeding large batches of data directly into a pre-warmed GPU memory state.
