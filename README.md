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
![Alt text for the image](assets/jupyternotebookPerformance.png)

### ⏳ The Interruption (and the Pivot)
Right in the middle of wrestling with these memory bottlenecks, university exams hit. I had to shelve the OpenCL experiments to focus on a massive 5-day intensive study plan, specifically grinding through Theory of Computation and Machine Learning.

Ironically, stepping away from the code was exactly what the project needed.

During my study breaks, I started playing around with FastAPI to get a better grip on backend architecture. As I got comfortable with FastAPI’s asynchronous routing and dependency injection, a lightbulb went off.

My OpenCL script was failing because it was structured like a one-off procedural script. It was recompiling the math kernels and rebuilding the GPU command queue every single time I hit "run."

What if I built an API where the GPU context was initialized exactly once? If I could tie the OpenCL environment to the lifecycle of a web server, the endpoints would simply act as funnels, feeding large batches of data directly into a pre-warmed GPU memory state.


### ⚙️ Optimized GPU Compute Architecture (Final Design)
When moving from a raw Jupyter notebook to a production-ready application, the biggest bottleneck isn't usually the math—it is state management and data validation. If the CPU wastes time improperly formatting data or repeatedly rebuilding the GPU environment, the parallel processing gains vanish completely.

To solve this, I wrapped the OpenCL logic inside a FastAPI backend, breaking the pipeline into four distinct, highly optimized layers.

1. The Pydantic Gatekeeper (arrayInfo.py)
You can't just blindly feed data into a GPU. If the array size is too small, the CPU overhead of transferring the data ruins the efficiency. If it's too large, you risk exceeding the RX 6500M's 4GB VRAM, causing an out-of-memory crash.

To prevent this, I used Pydantic models to enforce strict constraints before the request ever touches the compute logic.

Python
class InfoAcceptor(BaseModel):
    arraySize : int = Field(ge = 1000 , le = 100_000_000)
    operation : Operation
By setting ge=1000 and le=100_000_000, the API guarantees that the GPU is only invoked for workloads where it actually provides a benefit, while keeping the memory footprint safely within the hardware limits.

2. Pre-Compiling the C Kernels (pyopenclcompute.py)
This is where the real performance is unlocked. In OpenCL, math operations are written in C (called "kernels"). Compiling this C code into binary instructions for the GPU at runtime is incredibly slow.

Instead of recompiling every time a calculation is requested, the OpenCLCalculatorService class actively scans the hardware platforms, finds the specific GPU (the RX 6500M), and pre-compiles the kernels during the class initialization:

Python
def __init__(self):
    self.ctx, self.queue = self._get_rx6500m_queue()
    
    # Kernels are compiled into memory immediately
    self.kernels = {
        'add': ElementwiseKernel(self.ctx, "float *a, float *b, float *c", "c[i] = a[i] + b[i]", "add_kernel"),
        'sub': ElementwiseKernel(self.ctx, "float *a, float *b, float *c", "c[i] = a[i] - b[i]", "sub_kernel"),
        'mul': ElementwiseKernel(self.ctx, "float *a, float *b, float *c", "c[i] = a[i] * b[i]", "mul_kernel")
    }
This ensures the math instructions are already sitting in the GPU's memory, waiting for data. Furthermore, when calculate() is called, the NumPy arrays are strictly formatted as contiguous 32-bit floats (np.ascontiguousarray(arr, dtype=np.float32)) to ensure smooth memory transfer to the device buffers.

3. State Management with FastAPI Lifespan (main.py)
If we initialized the OpenCLCalculatorService inside the route handler, the API would reconnect to the GPU and recompile the kernels on every single API call. That overhead would completely destroy the OpenCL scaling advantage.

Instead, I used FastAPI's lifespan context manager. This ensures the GPU service is instantiated exactly once when the server boots up, attached globally to the application state, and gracefully destroyed when the server shuts down.

Python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initializes the GPU context and compiles kernels globally
    app.state.gpu_service = OpenCLCalculatorService()
    yield
    # Cleanup memory on shutdown
    del app.state.gpu_service

app = FastAPI(lifespan=lifespan)
4. The Execution Pipeline (calculate.py)
With the foundation built, the actual API endpoint is incredibly lean. When a POST request hits /OpenCLCompute/calculateArray, the pipeline executes in three smooth steps:

Data Generation: It generates two contiguous 32-bit float NumPy arrays based on the requested, validated size.

CPU Compute: It passes the arrays to the numpycompute.py service for a baseline time measurement.

GPU Compute: It retrieves the pre-warmed gpu_service from the app state, pushes the arrays into the GPU, executes the pre-compiled operation, and pulls the results back.

Python
@arrayCalclate.post('/OpenCLCompute/calculateArray')
async def calculateArray(request: Request, info: InfoAcceptor = Body(...)):
    # 1. Data generation
    arr1 = np.random.rand(info.arraySize).astype(np.float32)
    arr2 = np.random.rand(info.arraySize).astype(np.float32)

    # 2. Accessing the pre-warmed GPU context
    gpu_service = request.app.state.gpu_service

    # 3. Benchmarking both pipelines
    numpy_result, numpy_time = computeNP(arr1, arr2, info.operation.value)
    opencl_results, opencl_time = gpu_service.calculate(arr1, arr2, info.operation.value)

    return {
        "numpy_time": float(numpy_time),
        "opencl_time": float(opencl_time) 
    }
By isolating the GPU context initialization from the actual request cycle, the overhead drops to nearly zero. The OpenCL execution time reported by the API is now a true reflection of the hardware's math capabilities, completely unburdened by setup latency.

Here is an good diagram for understanding

![Alt text for the image](assets/pyopenclbasics.png)
