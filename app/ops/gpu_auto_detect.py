#!/usr/bin/env python3
"""
GPU Auto-Detection Module - SwapMaster V1 (Native)
Converted from gpu_auto_detect.sh to Python for cross-platform support.
"""
import os
import subprocess
import sys


def detect_gpu():
    """Detect GPU and set appropriate environment variables."""
    gpu_detected = False
    gpu_name = "None"
    gpu_vram_mb = 0

    # Try nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            parts = result.stdout.strip().split(",")
            gpu_name = parts[0].strip()
            vram_str = parts[1].strip().replace("MiB", "").replace("MB", "").strip()
            gpu_vram_mb = int(vram_str)
            print(f"[GPU-DETECT] Found: {gpu_name} | VRAM: {gpu_vram_mb}MB")

            # Check CUDA availability in onnxruntime
            try:
                import onnxruntime as ort
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    gpu_detected = True
                else:
                    print("[GPU-DETECT] WARNING: nvidia-smi found GPU but CUDAExecutionProvider unavailable")
            except ImportError:
                print("[GPU-DETECT] WARNING: onnxruntime not installed")
    except FileNotFoundError:
        print("[GPU-DETECT] nvidia-smi not found")
    except Exception as e:
        print(f"[GPU-DETECT] GPU detection error: {e}")

    if gpu_detected:
        os.environ["EXECUTION_PROVIDER"] = "cuda"
        os.environ["GPU_ONLY_MODE"] = "1"
        os.environ["OUTPUT_VIDEO_ENCODER"] = "h264_nvenc"

        if gpu_vram_mb >= 20000:
            os.environ["FACE_SWAPPER_MODEL"] = "hyperswap_1a_256"
            os.environ["EXECUTION_THREAD_COUNT"] = "8"
            print(f"[GPU-DETECT] High VRAM ({gpu_vram_mb}MB) -> hyperswap_1a_256, 8 threads")
        elif gpu_vram_mb >= 12000:
            os.environ["FACE_SWAPPER_MODEL"] = "hyperswap_1a_256"
            os.environ["EXECUTION_THREAD_COUNT"] = "4"
            print(f"[GPU-DETECT] Standard VRAM ({gpu_vram_mb}MB) -> hyperswap_1a_256, 4 threads")
        elif gpu_vram_mb >= 6000:
            os.environ["FACE_SWAPPER_MODEL"] = "inswapper_128_fp16"
            os.environ["EXECUTION_THREAD_COUNT"] = "4"
            print(f"[GPU-DETECT] Mid VRAM ({gpu_vram_mb}MB) -> inswapper_128_fp16, 4 threads")
        else:
            os.environ["FACE_SWAPPER_MODEL"] = "inswapper_128_fp16"
            os.environ["EXECUTION_THREAD_COUNT"] = "2"
            os.environ["ENABLE_FACE_ENHANCER"] = "0"
            print(f"[GPU-DETECT] Low VRAM ({gpu_vram_mb}MB) -> inswapper_128_fp16, enhancer OFF")

        # Set CUDA library paths
        _set_cuda_library_path()
    else:
        os.environ["EXECUTION_PROVIDER"] = "cpu"
        os.environ["GPU_ONLY_MODE"] = "0"
        os.environ["OUTPUT_VIDEO_ENCODER"] = "libx264"
        os.environ["FACE_SWAPPER_MODEL"] = "inswapper_128_fp16"
        os.environ["EXECUTION_THREAD_COUNT"] = "4"
        print("[GPU-DETECT] No GPU -> CPU mode")

    os.environ["GPU_DETECTED_NAME"] = gpu_name
    os.environ["GPU_DETECTED_VRAM_MB"] = str(gpu_vram_mb)
    print(f"[GPU-DETECT] Final: PROVIDER={os.environ['EXECUTION_PROVIDER']} | "
          f"MODEL={os.environ['FACE_SWAPPER_MODEL']} | THREADS={os.environ['EXECUTION_THREAD_COUNT']}")

    return gpu_detected, gpu_name, gpu_vram_mb


def _set_cuda_library_path():
    """Set LD_LIBRARY_PATH for CUDA libraries."""
    try:
        import sys as _sys
        from pathlib import Path

        # Find nvidia packages in current Python environment
        python_dir = Path(_sys.executable).parent
        nvidia_base = python_dir.parent / "lib" / "python3.12" / "site-packages" / "nvidia"

        if not nvidia_base.is_dir():
            # Try alternative: conda env
            nvidia_base = Path("/opt/conda/envs/cloudspace/lib/python3.12/site-packages/nvidia")
            if not nvidia_base.is_dir():
                return

        nvidia_libs = []
        for sub in ["cublas", "cudnn", "cuda_runtime", "cufft", "curand",
                     "cusolver", "cusparse", "nccl", "nvjitlink"]:
            lib_path = nvidia_base / sub / "lib"
            if lib_path.is_dir():
                nvidia_libs.append(str(lib_path))

        if nvidia_libs:
            current_ld = os.environ.get("LD_LIBRARY_PATH", "")
            cuda_paths = [
                "/usr/local/cuda/targets/x86_64-linux/lib",
                "/usr/lib/x86_64-linux-gnu"
            ]
            new_ld = ":".join(nvidia_libs + cuda_paths)
            if current_ld:
                new_ld += f":{current_ld}"
            os.environ["LD_LIBRARY_PATH"] = new_ld
            print("[GPU-DETECT] LD_LIBRARY_PATH set for CUDA")
    except Exception as e:
        print(f"[GPU-DETECT] Warning: Could not set LD_LIBRARY_PATH: {e}")


if __name__ == "__main__":
    detect_gpu()
