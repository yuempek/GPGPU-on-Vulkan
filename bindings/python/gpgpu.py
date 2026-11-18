"""
GPGPU.VULKAN Python Binding
High-Performance, Developer-Friendly GPGPU Compute Framework on Vulkan.
Uses pure ctypes for zero external build dependencies. Seamlessly integrates with NumPy if present.
"""

import ctypes
import os
import sys
from typing import Optional, List, Union, Callable

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# ==============================================================================
# Shared Library Discovery & Loader
# ==============================================================================

def _find_library() -> str:
    # 1. Explicit environment variable overrides
    for env_var in ("GPGPU_VULKAN_LIB", "GPGPU_VULKAN_SO", "GPGPU_VULKAN_DLL"):
        env_path = os.environ.get(env_var)
        if env_path and os.path.exists(env_path):
            return os.path.normpath(env_path)

    base_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. Determine OS platform and target shared library filenames
    is_windows = sys.platform.startswith("win")
    is_darwin = sys.platform == "darwin"
    is_linux = not is_windows and not is_darwin

    if is_windows:
        lib_names = ["gpgpu_vulkan.dll", "libgpgpu_vulkan.dll"]
    elif is_darwin:
        lib_names = ["libgpgpu_vulkan.dylib", "gpgpu_vulkan.dylib"]
    else:
        # Linux / POSIX - strictly search for ELF .so, never for Windows .dll
        lib_names = ["libgpgpu_vulkan.so", "gpgpu_vulkan.so"]

    # 3. Determine CPU architecture subdirectories
    import platform
    machine = platform.machine().lower()

    if is_windows:
        arch_dirs = ["windows-x64", "x64", "x86_64", "win64"]
    elif is_darwin:
        if "arm" in machine or "aarch" in machine:
            arch_dirs = ["macos-arm64", "arm64"]
        else:
            arch_dirs = ["macos-x86_64", "x86_64"]
    else:
        # Linux: Distinguish Raspberry Pi ARM (aarch64 / armv7l) from PC (x86_64 / x86)
        if machine in ("aarch64", "arm64"):
            arch_dirs = ["linux-aarch64", "aarch64", "arm64", "linux-rpi-arm64", "rpi-arm64"]
        elif machine.startswith("arm") or machine.startswith("aarch32"):
            arch_dirs = ["linux-armhf", "linux-armv7l", "armv7l", "armhf", "arm"]
        elif machine in ("x86_64", "amd64"):
            arch_dirs = ["linux-x86_64", "x86_64", "x64"]
        elif machine in ("i386", "i686", "x86"):
            arch_dirs = ["linux-x86", "x86", "i686", "i386"]
        else:
            arch_dirs = [f"linux-{machine}", machine]

    # 4. Search locations in priority order
    search_dirs = []

    # Priority A: Architecture-specific directory in deploy/lib or lib
    for ad in arch_dirs:
        search_dirs.append(os.path.join(base_dir, "..", "..", "lib", ad))
        search_dirs.append(os.path.join(base_dir, "..", "..", "deploy", "lib", ad))
        search_dirs.append(os.path.join(base_dir, "..", "lib", ad))
        search_dirs.append(os.path.join(base_dir, "lib", ad))

    # Priority B: Standard deploy/lib and lib directories
    search_dirs.append(os.path.join(base_dir, "..", "..", "lib"))
    search_dirs.append(os.path.join(base_dir, "..", "..", "deploy", "lib"))
    search_dirs.append(os.path.join(base_dir, "..", "lib"))
    search_dirs.append(os.path.join(base_dir, "lib"))

    # Priority C: Build directories (for developers working from repository root)
    search_dirs.append(os.path.join(base_dir, "..", "..", "build", "release"))
    search_dirs.append(os.path.join(base_dir, "..", "..", "build", "linux-release"))
    search_dirs.append(os.path.join(base_dir, "..", "..", "build", "rpi-release"))
    search_dirs.append(os.path.join(base_dir, "..", "..", "build", "debug"))
    search_dirs.append(os.path.join(base_dir, "..", "..", "build"))
    search_dirs.append(base_dir)

    for d in search_dirs:
        for name in lib_names:
            p = os.path.join(d, name)
            if os.path.exists(p):
                return os.path.normpath(p)

    # 5. Fallback to system dynamic library loader
    try:
        import ctypes.util
        sys_lib = ctypes.util.find_library("gpgpu_vulkan")
        if sys_lib:
            return sys_lib
    except Exception:
        pass

    return lib_names[0]

_LIB_PATH = _find_library()
try:
    _lib = ctypes.CDLL(_LIB_PATH)
except Exception as e:
    import platform
    _m = platform.machine()
    _arch_info = {
        "aarch64": "Raspberry Pi 4/5 (ARM64 / AArch64)",
        "arm64":   "Raspberry Pi 4/5 (ARM64 / AArch64)",
        "armv7l":  "Raspberry Pi 2/3/4 (ARMv7 32-bit)",
        "x86_64":  "Linux x86_64 (Intel/AMD 64-bit)",
        "amd64":   "Linux x86_64 (Intel/AMD 64-bit)",
        "i686":    "Linux x86 (Intel/AMD 32-bit)",
        "i386":    "Linux x86 (Intel/AMD 32-bit)",
    }.get(_m.lower(), f"Machine: {_m}")

    raise RuntimeError(
        f"\n{'='*78}\n"
        f"GPGPU.VULKAN — Shared Library Load Error\n"
        f"{'='*78}\n"
        f"Could not load shared library from: '{_LIB_PATH}'\n"
        f"Platform: {sys.platform} | Architecture: {_m} ({_arch_info})\n"
        f"Error details: {e}\n\n"
        f"Diagnosis & Fix:\n"
        f"  1. Architecture mismatch / missing native binary:\n"
        f"     - Windows '.dll' cannot be loaded on Linux ('invalid ELF header').\n"
        f"     - Intel/AMD x86_64 '.so' cannot be loaded on ARM Raspberry Pi ('wrong ELF class').\n"
        f"     - ARM Raspberry Pi requires a native ARM-compiled 'libgpgpu_vulkan.so'.\n\n"
        f"  2. Build native .so for this machine in 1 command:\n"
        f"     Run:  ./build.sh  (or bash deploy/build_linux.sh)\n\n"
        f"  3. Or explicitly specify the library path:\n"
        f"     export GPGPU_VULKAN_LIB=/path/to/libgpgpu_vulkan.so\n"
        f"{'='*78}"
    ) from e

# ==============================================================================
# C99 ABI Declarations
# ==============================================================================

GPGPU_DTYPE_FLOAT32 = 0
GPGPU_DTYPE_FLOAT64 = 1
GPGPU_DTYPE_INT32   = 2
GPGPU_DTYPE_UINT32  = 3
GPGPU_DTYPE_INT16   = 4
GPGPU_DTYPE_UINT16  = 5
GPGPU_DTYPE_INT8    = 6
GPGPU_DTYPE_UINT8   = 7
GPGPU_DTYPE_VEC2    = 8
GPGPU_DTYPE_VEC3    = 9
GPGPU_DTYPE_VEC4    = 10

GPGPU_INTERP_NONE    = 0
GPGPU_INTERP_NEAREST = 1
GPGPU_INTERP_LINEAR  = 2
GPGPU_INTERP_CUBIC   = 3

def _parse_interp(method: str) -> int:
    m = str(method).lower().strip()
    if m in ("linear", "lineer", "bilinear"):
        return GPGPU_INTERP_LINEAR
    elif m in ("cubic", "bicubic", "kubik"):
        return GPGPU_INTERP_CUBIC
    elif m in ("nearest", "point"):
        return GPGPU_INTERP_NEAREST
    return GPGPU_INTERP_NONE

# Function signatures
_lib.gpgpu_get_device_count.argtypes = []
_lib.gpgpu_get_device_count.restype = ctypes.c_int

_lib.gpgpu_get_device_name.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_size_t]
_lib.gpgpu_get_device_name.restype = ctypes.c_int

_lib.gpgpu_create_auto.argtypes = []
_lib.gpgpu_create_auto.restype = ctypes.c_void_p

_lib.gpgpu_create_by_index.argtypes = [ctypes.c_int]
_lib.gpgpu_create_by_index.restype = ctypes.c_void_p

_lib.gpgpu_create_by_name.argtypes = [ctypes.c_char_p]
_lib.gpgpu_create_by_name.restype = ctypes.c_void_p

_lib.gpgpu_dispose.argtypes = [ctypes.c_void_p]
_lib.gpgpu_dispose.restype = None

_lib.gpgpu_destroy.argtypes = [ctypes.c_void_p]
_lib.gpgpu_destroy.restype = None

_lib.gpgpu_create_data.argtypes = [
    ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int
]
_lib.gpgpu_create_data.restype = ctypes.c_void_p

_lib.gpgpu_data_set.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_lib.gpgpu_data_set.restype = ctypes.c_int

_lib.gpgpu_data_get_element_count.argtypes = [ctypes.c_void_p]
_lib.gpgpu_data_get_element_count.restype = ctypes.c_size_t

_lib.gpgpu_data_get_dims.argtypes = [
    ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32)
]
_lib.gpgpu_data_get_dims.restype = ctypes.c_int

_lib.gpgpu_data_get.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
_lib.gpgpu_data_get.restype = ctypes.c_int

_lib.gpgpu_data_dispose.argtypes = [ctypes.c_void_p]
_lib.gpgpu_data_dispose.restype = None

_lib.gpgpu_kernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.gpgpu_kernel.restype = ctypes.c_int

_lib.gpgpu_set_kernel.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.gpgpu_set_kernel.restype = ctypes.c_int

_lib.gpgpu_run.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
_lib.gpgpu_run.restype = ctypes.c_int

_lib.gpgpu_run_async.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32]
_lib.gpgpu_run_async.restype = ctypes.c_uint64

_lib.gpgpu_wait.argtypes = [ctypes.c_void_p]
_lib.gpgpu_wait.restype = None

CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
_lib.gpgpu_on_done.argtypes = [ctypes.c_void_p, CALLBACK_TYPE, ctypes.c_void_p]
_lib.gpgpu_on_done.restype = None

_lib.gpgpu_has_error.argtypes = [ctypes.c_void_p]
_lib.gpgpu_has_error.restype = ctypes.c_char_p

_lib.gpgpu_clear_error.argtypes = [ctypes.c_void_p]
_lib.gpgpu_clear_error.restype = None

_lib.gpgpu_last_elapsed_ms.argtypes = [ctypes.c_void_p]
_lib.gpgpu_last_elapsed_ms.restype = ctypes.c_double

# ==============================================================================
# Python Wrapper Classes
# ==============================================================================

class Data:
    """Represents a GPU memory buffer bound by name in GLSL kernels."""

    def __init__(self, handle: int, dtype: int = GPGPU_DTYPE_FLOAT32):
        self._handle = handle
        self._dtype = dtype

    @property
    def handle(self) -> int:
        return self._handle

    def set(self, data: Union[List[float], "np.ndarray"]) -> None:
        if not self._handle:
            raise RuntimeError("Data handle is null or has been disposed.")

        if HAS_NUMPY and isinstance(data, np.ndarray):
            arr = np.ascontiguousarray(data, dtype=np.float32)
            ptr = arr.ctypes.data_as(ctypes.c_void_p)
            res = _lib.gpgpu_data_set(self._handle, ptr, arr.size)
        else:
            flat = list(data)
            c_arr = (ctypes.c_float * len(flat))(*flat)
            res = _lib.gpgpu_data_set(self._handle, ctypes.cast(c_arr, ctypes.c_void_p), len(flat))

        if res != 0:
            raise RuntimeError("Failed to upload data to GPU buffer.")

    def get(self) -> Union[List[float], "np.ndarray"]:
        if not self._handle:
            raise RuntimeError("Data handle is null or has been disposed.")

        count = _lib.gpgpu_data_get_element_count(self._handle)
        if count == 0:
            return np.array([], dtype=np.float32) if HAS_NUMPY else []

        # Get buffer dimensions
        dx, dy, dz = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
        _lib.gpgpu_data_get_dims(self._handle, ctypes.byref(dx), ctypes.byref(dy), ctypes.byref(dz))

        if HAS_NUMPY:
            out = np.empty(count, dtype=np.float32)
            ptr = out.ctypes.data_as(ctypes.c_void_p)
            res = _lib.gpgpu_data_get(self._handle, ptr, count)
            if res != 0:
                raise RuntimeError("Failed to read back data from GPU buffer.")

            if dz.value > 1:
                return out.reshape((dz.value, dy.value, dx.value))
            elif dy.value > 1:
                return out.reshape((dy.value, dx.value))
            return out
        else:
            c_arr = (ctypes.c_float * count)()
            res = _lib.gpgpu_data_get(self._handle, ctypes.cast(c_arr, ctypes.c_void_p), count)
            if res != 0:
                raise RuntimeError("Failed to read back data from GPU buffer.")
            return list(c_arr)

    def element_count(self) -> int:
        return _lib.gpgpu_data_get_element_count(self._handle) if self._handle else 0

    def dimensions(self) -> tuple:
        if not self._handle:
            return (0, 0, 0)
        dx, dy, dz = ctypes.c_uint32(), ctypes.c_uint32(), ctypes.c_uint32()
        _lib.gpgpu_data_get_dims(self._handle, ctypes.byref(dx), ctypes.byref(dy), ctypes.byref(dz))
        return (dx.value, dy.value, dz.value)

    def dispose(self) -> None:
        if self._handle:
            _lib.gpgpu_data_dispose(self._handle)
            self._handle = None

    def __del__(self):
        self.dispose()


class ExecutionTicket:
    """Handle to an in-flight asynchronous GPU compute dispatch."""

    def __init__(self, gpu_ref, ticket_id: int):
        self._gpu = gpu_ref
        self._ticket_id = ticket_id

    @property
    def id(self) -> int:
        return self._ticket_id

    def wait(self) -> None:
        if self._gpu:
            self._gpu.wait()

    def on_done(self, callback: Callable[[], None]) -> "ExecutionTicket":
        if self._gpu:
            self._gpu.on_done(callback)
        return self


class GPU:
    """
    Main GPGPU.VULKAN compute context.
    Matches the pseudo-code API 1:1:
        gpu = GPU(1 or 'RTX 3060')
        remoteData = gpu.createData('varNameIn', localInputData)
        remoteOutData = gpu.createData('output')
        gpu.kernel('kernel(coord){ outputData[coord] = cos(varNameIn[coord]) }')
        gpu.run(800, 600)
        localOutputData = remoteOutData.get()
        gpu.dispose()
    """

    def __init__(self, device: Optional[Union[int, str]] = None):
        if device is None:
            self._ctx = _lib.gpgpu_create_auto()
        elif isinstance(device, int):
            self._ctx = _lib.gpgpu_create_by_index(device)
        elif isinstance(device, str):
            self._ctx = _lib.gpgpu_create_by_name(device.encode("utf-8"))
        else:
            raise TypeError("Device selector must be an integer index or substring name.")

        if not self._ctx:
            raise RuntimeError("Failed to create GPGPU.VULKAN context.")

        self._bound_data = []
        self._active_callback_ref = None

    @staticmethod
    def devices() -> List[str]:
        count = _lib.gpgpu_get_device_count()
        devs = []
        buf = ctypes.create_string_buffer(256)
        for i in range(count):
            if _lib.gpgpu_get_device_name(i, buf, 256) == 0:
                devs.append(buf.value.decode("utf-8"))
        return devs

    def createData(
        self,
        var_name: str,
        local_data: Optional[Union[List[float], "np.ndarray"]] = None,
        interpolation: str = "none",
    ) -> Data:
        if not self._ctx:
            raise RuntimeError("GPU context has been disposed.")

        interp_mode = _parse_interp(interpolation)
        dtype = GPGPU_DTYPE_FLOAT32

        if local_data is not None:
            if HAS_NUMPY and isinstance(local_data, np.ndarray):
                arr = np.ascontiguousarray(local_data, dtype=np.float32)
                ptr = arr.ctypes.data_as(ctypes.c_void_p)
                count = arr.size
            else:
                flat = list(local_data)
                c_arr = (ctypes.c_float * len(flat))(*flat)
                ptr = ctypes.cast(c_arr, ctypes.c_void_p)
                count = len(flat)

            handle = _lib.gpgpu_create_data(
                self._ctx, var_name.encode("utf-8"), ptr, count, dtype, interp_mode
            )
        else:
            handle = _lib.gpgpu_create_data(
                self._ctx, var_name.encode("utf-8"), None, 0, dtype, interp_mode
            )

        if not handle:
            err = self.hasError()
            raise RuntimeError(f"Failed to create Data '{var_name}': {err}")

        d = Data(handle, dtype)
        self._bound_data.append(d)
        return d

    # snake_case alias
    create_data = createData

    def kernel(self, code: str) -> None:
        if not self._ctx:
            raise RuntimeError("GPU context has been disposed.")
        res = _lib.gpgpu_set_kernel(self._ctx, code.encode("utf-8"))
        if res != 0:
            err = self.hasError()
            raise RuntimeError(f"Kernel error: {err}")

    def run(self, width: int, height: int = 1, depth: int = 1) -> None:
        if not self._ctx:
            raise RuntimeError("GPU context has been disposed.")
        res = _lib.gpgpu_run(self._ctx, width, height, depth)
        if res != 0:
            err = self.hasError()
            raise RuntimeError(f"Run error: {err}")

    def runAsync(self, width: int, height: int = 1, depth: int = 1) -> ExecutionTicket:
        if not self._ctx:
            raise RuntimeError("GPU context has been disposed.")
        ticket_id = _lib.gpgpu_run_async(self._ctx, width, height, depth)
        if ticket_id == 0:
            err = self.hasError()
            raise RuntimeError(f"RunAsync error: {err}")
        return ExecutionTicket(self, ticket_id)

    run_async = runAsync

    def wait(self) -> None:
        if self._ctx:
            _lib.gpgpu_wait(self._ctx)

    def onDone(self, callback: Callable[[], None]) -> None:
        if not self._ctx:
            return

        def _c_callback(user_data):
            callback()

        self._active_callback_ref = CALLBACK_TYPE(_c_callback)
        _lib.gpgpu_on_done(self._ctx, self._active_callback_ref, None)

    on_done = onDone

    def hasError(self) -> Optional[str]:
        if not self._ctx:
            return "GPU context is disposed"
        err_ptr = _lib.gpgpu_has_error(self._ctx)
        if err_ptr:
            return err_ptr.decode("utf-8")
        return None

    has_error = hasError

    def clearError(self) -> None:
        if self._ctx:
            _lib.gpgpu_clear_error(self._ctx)

    clear_error = clearError

    def lastKernelTimeMs(self) -> float:
        return _lib.gpgpu_last_elapsed_ms(self._ctx) if self._ctx else 0.0

    last_kernel_time_ms = lastKernelTimeMs

    def dispose(self) -> None:
        for d in self._bound_data:
            d.dispose()
        self._bound_data.clear()

        if self._ctx:
            _lib.gpgpu_destroy(self._ctx)
            self._ctx = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.dispose()

    def __del__(self):
        self.dispose()
