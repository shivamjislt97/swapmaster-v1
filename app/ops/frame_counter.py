"""
Frame counter for FaceFusion pipeline.

FaceFusion extracts ALL frames first, then overwrites them in-place during processing.
So disk frame count always equals total_frames — it cannot measure progress.

Ground truth for processed frames comes from active_job_state.json (done_frames),
which is written by the worker process directly from FaceFusion's tqdm callback.
"""

import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_frame_count_from_active_state(state: dict) -> dict:
    """
    Returns progress from active_job_state.json.
    done_frames is written by the worker directly from FaceFusion's progress callback.
    """
    done   = int(state.get("done_frames", state.get("frames_done", 0)))
    total  = int(state.get("total_frames", 0))
    pct    = round(done / total * 100, 1) if total > 0 else int(state.get("progress", 0))
    return {
        "processed_frames": done,
        "total_frames":     total,
        "percent":          pct,
    }
