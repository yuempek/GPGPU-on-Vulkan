import sys
import os
import math

# Add bindings/python to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "bindings", "python"))

from gpgpu import GPU

def test_python_binding():
    print("--> Testing GPU.devices()...")
    devices = GPU.devices()
    print(f"Available devices: {devices}")
    assert len(devices) > 0, "No devices found"

    print("--> Testing GPU() initialization...")
    gpu = GPU(0)
    assert gpu.hasError() is None

    print("--> Testing pseudo-code workflow in Python...")
    # User's pseudo-code in Python:
    # remoteData = gpu.createData('varNameIn', localInputData)
    # remoteOutData = gpu.createData('output')
    # gpu.kernel('kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }')
    # gpu.run(800, 600)
    # localOutputData = remoteOutData.get()

    N = 1000
    local_in = [float(i) * 0.01 for i in range(N)]
    remote_data = gpu.createData("varNameIn", local_in)
    remote_out = gpu.createData("outputData")

    gpu.kernel("kernel(coord){ outputData[coord] = cos(varNameIn[coord]); }")
    gpu.run(N)

    local_out = remote_out.get()
    assert len(local_out) == N, f"Expected {N} elements, got {len(local_out)}"

    for i in range(N):
        expected = math.cos(local_in[i])
        actual = local_out[i]
        diff = abs(actual - expected)
        assert diff < 5e-5, f"Mismatch at index {i}: actual={actual}, expected={expected}"

    print(f"1D Cosine test passed! 1000 elements verified within 5e-5 tolerance.")

    # 2D execution test (800 x 600)
    print("--> Testing 2D execution (800 x 600)...")
    out_2d = gpu.createData("matrixOut")
    gpu.kernel("kernel(coord){ matrixOut[coord] = float(coord.x + coord.y); }")
    gpu.run(800, 600)
    res_2d = out_2d.get()
    assert len(res_2d) == 800 * 600
    print(f"2D execution passed! Element [0]={res_2d[0]}, [800*599+799]={res_2d[-1]}")

    elapsed_ms = gpu.lastKernelTimeMs()
    print(f"Last kernel execution time: {elapsed_ms:.3f} ms")

    gpu.dispose()
    print("--> Python binding test successfully completed!")

if __name__ == "__main__":
    test_python_binding()
