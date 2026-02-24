import pyopencl as cl
import pyopencl.array as cl_array
from pyopencl.elementwise import ElementwiseKernel
import numpy as np

class OpenCLCalculatorService:
    def __init__(self):
        self.ctx, self.queue = self._get_rx6500m_queue()
        
        self.kernels = {
            'add': ElementwiseKernel(self.ctx, "float *a, float *b, float *c", "c[i] = a[i] + b[i]", "add_kernel"),
            'sub': ElementwiseKernel(self.ctx, "float *a, float *b, float *c", "c[i] = a[i] - b[i]", "sub_kernel"),
            'mul': ElementwiseKernel(self.ctx, "float *a, float *b, float *c", "c[i] = a[i] * b[i]", "mul_kernel")
        }

    def _get_rx6500m_queue(self):
        for platform in cl.get_platforms():
            try:
                for gpu in platform.get_devices(device_type=cl.device_type.GPU):
                    if "6500" in gpu.name:
                        return cl.Context([gpu]), cl.CommandQueue(gpu, properties=cl.command_queue_properties.PROFILING_ENABLE)
            except cl.LogicError:
                continue
                
        fallback_ctx = cl.create_some_context(interactive=False)
        return fallback_ctx, cl.CommandQueue(fallback_ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

    def calculate(self, arr1: np.ndarray, arr2: np.ndarray, operation: str):
        if operation not in self.kernels:
            raise ValueError(f"Unsupported operation: {operation}")

        arr1_f32 = np.ascontiguousarray(arr1, dtype=np.float32)
        arr2_f32 = np.ascontiguousarray(arr2, dtype=np.float32)

        arr1_g = cl_array.to_device(self.queue, arr1_f32)
        arr2_g = cl_array.to_device(self.queue, arr2_f32)
        c_g = cl_array.empty_like(arr1_g)

        kernel_func = self.kernels[operation]
        
        exec_event = kernel_func(arr1_g, arr2_g, c_g)
        exec_event.wait()
        
        elapsed_time = (exec_event.profile.end - exec_event.profile.start) * 1e-9
        c_cpu = c_g.get()

        return c_cpu, elapsed_time
