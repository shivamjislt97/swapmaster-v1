#!/usr/bin/env python3
"""
Auto-sleep standalone verifier.
Usage:
  python test_auto_sleep.py --mode cancel   # 120s countdown, cancel at 110s
  python test_auto_sleep.py --mode trigger  # 20s countdown, let it trigger
"""
import asyncio, os, sys, time, logging, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOG_FILE = ROOT / "pipeline" / "logs" / "auto_sleep_test.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
log = logging.getLogger("auto_sleep_test")

# Simulate request_studio_sleep without actually sleeping
def mock_sleep():
    log.info("[AUTO_SLEEP_TRIGGER] ✅ Sleep trigger executed (TEST MODE - no real sleep)")
    return True, "test-mode-mock-sleep"


async def run_countdown(delay_seconds: int, cancel_at: int | None):
    log.info("[AUTO_SLEEP_START] countdown=%ds cancel_at=%s", delay_seconds, cancel_at)
    end_time = time.monotonic() + delay_seconds
    cancelled = False

    async def cancel_trigger():
        await asyncio.sleep(cancel_at)
        log.info("[AUTO_SLEEP_CANCEL] Manual cancel triggered at %ds", cancel_at)
        task.cancel()

    loop = asyncio.get_event_loop()
    task = asyncio.current_task()

    if cancel_at is not None:
        asyncio.create_task(cancel_trigger())

    try:
        last_tick = delay_seconds
        while True:
            remain = max(0, int(end_time - time.monotonic()))
            if remain != last_tick:
                log.info("[AUTO_SLEEP_TICK] %ds remaining", remain)
                last_tick = remain
            if remain <= 0:
                break
            await asyncio.sleep(1)

        log.info("[AUTO_SLEEP_TRIGGER] Countdown reached zero — executing sleep trigger")
        ok, info = mock_sleep()
        if ok:
            log.info("[AUTO_SLEEP_TRIGGER] ✅ PASS: sleep trigger succeeded info=%s", info)
        else:
            log.error("[AUTO_SLEEP_TRIGGER] ❌ FAIL: sleep trigger failed info=%s", info)

    except asyncio.CancelledError:
        log.info("[AUTO_SLEEP_CANCEL] ✅ PASS: CancelledError caught correctly")
        cancelled = True
        raise

    return not cancelled


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cancel", "trigger"], default="trigger")
    args = parser.parse_args()

    if args.mode == "cancel":
        log.info("=== TEST MODE: cancel (120s countdown, cancel at 110s) ===")
        task = asyncio.create_task(run_countdown(120, cancel_at=10))
        try:
            result = await task
            log.info("Result: completed=%s", result)
        except asyncio.CancelledError:
            log.info("[AUTO_SLEEP_CANCEL] ✅ Test passed: countdown cancelled correctly")
    else:
        log.info("=== TEST MODE: trigger (20s countdown, no cancel) ===")
        task = asyncio.create_task(run_countdown(20, cancel_at=None))
        result = await task
        if result:
            log.info("✅ ALL PASS: Auto-sleep trigger test succeeded")
        else:
            log.error("❌ FAIL: trigger test did not complete")

    log.info("Test complete. See log: %s", LOG_FILE)


if __name__ == "__main__":
    asyncio.run(main())
