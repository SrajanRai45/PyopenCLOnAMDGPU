from fastapi import APIRouter, Body
from src.models.arrayInfo import InfoAcceptor
from src.services.numpycompute import computeNP
from src.services.pyopenclcompute import openclCalculate
import numpy as np

arrayCalclate = APIRouter()

@arrayCalclate.post('/OpenCLCompute/calculateArray')
async def calculateArray(info: InfoAcceptor = Body(...)):
    arr1 = np.random.rand(info.arraySize).astype(np.float32)
    arr2 = np.random.rand(info.arraySize).astype(np.float32)

    numpy_time = computeNP(arr1, arr2, info.operation.value)
    opencl_time = openclCalculate(arr1, arr2, info.operation.value)

    return {
        "numpy_time": numpy_time,
        "opencl_time": opencl_time
    }
