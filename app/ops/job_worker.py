#!/usr/bin/env python3
import argparse
import asyncio
import sys
import traceback
from pathlib import Path
from types import SimpleNamespace

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import bot as runtime_bot
from telegram import Bot


class WorkerContext(SimpleNamespace):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Runtime queue worker shim")
    parser.add_argument("--chat-id", required=True)
    parser.add_argument("--video-link", required=True)
    parser.add_argument("--face-link", default=None)
    parser.add_argument("--job-mode", default="direct")
    parser.add_argument("--gender-mode", default="all")
    parser.add_argument("--queue-job-id", type=int, default=0)
    parser.add_argument("--progress-seed-message-id", type=int, default=None)
    return parser.parse_args()


async def run_worker(args):
    runtime_bot.reload_runtime_credentials()
    token = str(runtime_bot.BOT_TOKEN or "").strip()
    if not token:
        print("job_worker error: BOT_TOKEN missing", flush=True)
        return 2

    bot_obj = Bot(token=token)
    context = WorkerContext(
        bot=bot_obj,
        application=None,
        user_data={},
        chat_data={},
        bot_data={},
    )

    chat_id = str(args.chat_id)
    print(
        f"job_worker start chat={chat_id} mode={args.job_mode} gender={args.gender_mode} "
        f"job_id={args.queue_job_id}",
        flush=True,
    )

    await runtime_bot.run_pipeline(
        context,
        chat_id,
        str(args.video_link),
        face_link=(str(args.face_link).strip() if args.face_link else None),
        job_mode=str(args.job_mode or "direct"),
        progress_seed_message_id=args.progress_seed_message_id,
        queue_job_id=int(args.queue_job_id or 0),
        gender_mode_override=str(args.gender_mode or "all"),
    )

    state = runtime_bot.job_status.get(chat_id, {}) or {}
    phase = str(state.get("phase") or "").strip().lower()
    print(f"job_worker done chat={chat_id} phase={phase}", flush=True)
    return 0 if phase == "completed" else 1


def main():
    args = parse_args()
    try:
        rc = asyncio.run(run_worker(args))
    except Exception:
        traceback.print_exc()
        rc = 2
    raise SystemExit(int(rc))


if __name__ == "__main__":
    main()
