#!/bin/bash
# GPU Auto-Detection Module — Swap Master V5 Pro
GPU_DETECTED=0; GPU_NAME="None"; GPU_VRAM_MB=0

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
    GPU_VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1 | sed 's/[^0-9]//g')
    echo "[GPU-DETECT] Found: $GPU_NAME | VRAM: ${GPU_VRAM_MB}MB"
    CUDA_OK=$(python3 -c "import onnxruntime as o; print('YES' if 'CUDAExecutionProvider' in o.get_available_providers() else 'NO')" 2>/dev/null || \
              /opt/conda/envs/cloudspace/bin/python3 -c "import onnxruntime as o; print('YES' if 'CUDAExecutionProvider' in o.get_available_providers() else 'NO')" 2>/dev/null)
    [ "$CUDA_OK" = "YES" ] && GPU_DETECTED=1 || echo "[GPU-DETECT] WARNING: nvidia-smi found GPU but CUDAExecutionProvider unavailable"
fi

if [ $GPU_DETECTED -eq 1 ]; then
    export EXECUTION_PROVIDER="cuda"
    export GPU_ONLY_MODE="1"
    export OUTPUT_VIDEO_ENCODER="h264_nvenc"
    if   [ "$GPU_VRAM_MB" -ge 20000 ]; then export FACE_SWAPPER_MODEL="hyperswap_1a_256"; export EXECUTION_THREAD_COUNT="8"; echo "[GPU-DETECT] High VRAM (${GPU_VRAM_MB}MB) → hyperswap_1a_256, 8 threads"
    elif [ "$GPU_VRAM_MB" -ge 12000 ]; then export FACE_SWAPPER_MODEL="hyperswap_1a_256"; export EXECUTION_THREAD_COUNT="4"; echo "[GPU-DETECT] Standard VRAM (${GPU_VRAM_MB}MB) → hyperswap_1a_256, 4 threads"
    elif [ "$GPU_VRAM_MB" -ge 6000 ];  then export FACE_SWAPPER_MODEL="inswapper_128_fp16"; export EXECUTION_THREAD_COUNT="4"; echo "[GPU-DETECT] Mid VRAM (${GPU_VRAM_MB}MB) → inswapper_128_fp16, 4 threads"
    else export FACE_SWAPPER_MODEL="inswapper_128_fp16"; export EXECUTION_THREAD_COUNT="2"; export ENABLE_FACE_ENHANCER="0"; echo "[GPU-DETECT] Low VRAM (${GPU_VRAM_MB}MB) → inswapper_128_fp16, enhancer OFF"
    fi
    # Build CUDA LD_LIBRARY_PATH from conda env nvidia packages
    NVIDIA_BASE="/opt/conda/envs/cloudspace/lib/python3.12/site-packages/nvidia"
    NVIDIA_LD=""
    if [ -d "$NVIDIA_BASE" ]; then
        for sub in cublas cudnn cuda_runtime cufft curand cusolver cusparse nccl nvjitlink; do
            p="$NVIDIA_BASE/$sub/lib"
            [ -d "$p" ] && NVIDIA_LD="$NVIDIA_LD:$p"
        done
    fi
    [ -n "$NVIDIA_LD" ] && export LD_LIBRARY_PATH="${NVIDIA_LD#:}:/usr/local/cuda/targets/x86_64-linux/lib:/usr/lib/x86_64-linux-gnu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" && echo "[GPU-DETECT] LD_LIBRARY_PATH set for CUDA"
else
    export EXECUTION_PROVIDER="cpu"; export GPU_ONLY_MODE="0"
    export OUTPUT_VIDEO_ENCODER="libx264"; export FACE_SWAPPER_MODEL="inswapper_128_fp16"
    export EXECUTION_THREAD_COUNT="4"; echo "[GPU-DETECT] No GPU → CPU mode"
fi

export GPU_DETECTED_NAME="$GPU_NAME"; export GPU_DETECTED_VRAM_MB="$GPU_VRAM_MB"
echo "[GPU-DETECT] Final: PROVIDER=$EXECUTION_PROVIDER | MODEL=$FACE_SWAPPER_MODEL | THREADS=$EXECUTION_THREAD_COUNT"
