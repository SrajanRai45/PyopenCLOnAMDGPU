import numpy as np
import time
from typing import List, Tuple

def computeNP(arr1: List[int], arr2: List[int], operation: str) -> Tuple[np.ndarray, float]:
    """
    Computes the specified operation on two lists using NumPy and returns
    the result plus the time taken for the calculation.
    """
    start_time = time.perf_counter()

    a = np.array(arr1, dtype=np.float32)
    b = np.array(arr2, dtype=np.float32)

    if operation == 'add':
        c = a + b
    elif operation == 'sub':
        c = a - b
    elif operation == 'mul':
        c = a * b
    else:
        raise ValueError("Unsupported operation")

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time

    return c, elapsed_time

# Example usage:
# result, duration = computeNP([1, 2, 3], [4, 5, 6])
# print(f"Result: {result}, Time: {duration:.6f}s")
