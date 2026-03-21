from fastapi import APIRouter, Body, Request
from src.models.arrayInfo import InfoAcceptor
from src.services.numpycompute import computeNP
import numpy as np

arrayCalclate = APIRouter()

@arrayCalclate.post('/OpenCLCompute/calculateArray')
async def calculateArray(request: Request, info: InfoAcceptor = Body(...)):
    arr1 = np.random.rand(info.arraySize).astype(np.float32)
    arr2 = np.random.rand(info.arraySize).astype(np.float32)

    gpu_service = request.app.state.gpu_service
    
    numpy_time = []
    opencl_time = []
    for i in range(0,100):
        numpy_time.append(float(computeNP(arr1, arr2, info.operation.value)))
    
    opencl_time , transfer_time = gpu_service.calculate(arr1, arr2, info.operation.value)

    return {
        "numpy_time": numpy_time,
        "opencl_time": opencl_time,
        "opencl_transfer_time": transfer_time
    }
