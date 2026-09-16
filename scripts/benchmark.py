"""Measure local inference/IPC overhead on synthetic images, without cameras.

This is a repeatable baseline, not person-tracking accuracy or motion latency.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cameras", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--quality", choices=("lite", "full", "heavy", "dwpose"), default="full")
    parser.add_argument("--image", type=Path, help="Repeat a saved person image to exercise refinement, without opening a camera")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.frames < 10:
        parser.error("--frames must be at least 10")
    import numpy as np
    from Lib.Workers import PoseWorker

    workers = []
    started = time.perf_counter()
    try:
        for _ in range(args.cameras):
            workers.append(PoseWorker(args.quality))
        startup_seconds = time.perf_counter() - started
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        if args.image:
            import cv2
            frame = cv2.imread(str(args.image))
            if frame is None:
                raise ValueError(f"Cannot read image: {args.image}")
        latencies = [[] for _ in workers]
        detections = [0 for _ in workers]

        def measure(index):
            for iteration in range(args.frames + 5):
                before = time.perf_counter()
                pose = workers[index].process(frame)
                elapsed = (time.perf_counter() - before) * 1000
                if iteration >= 5:
                    latencies[index].append(elapsed)
                    detections[index] += int(pose.detected)

        with ThreadPoolExecutor(max_workers=len(workers)) as executor:
            list(executor.map(measure, range(len(workers))))
        report = {
            "fixture": (f"Repeated saved image: {args.image}; inference plus local IPC only" if args.image else "1280x720 black BGR images; no person; inference plus local IPC only"),
            "quality": args.quality,
            "cameras": args.cameras,
            "frames_per_camera": args.frames,
            "startup_seconds": round(startup_seconds, 3),
            "python": sys.version.split()[0],
            "camera_results": [
                {"camera": index + 1, "detected_frames": detections[index], "p50_ms": round(statistics.median(samples), 3),
                 "p95_ms": round(float(np.percentile(samples, 95)), 3),
                 "p99_ms": round(float(np.percentile(samples, 99)), 3)}
                for index, samples in enumerate(latencies)
            ],
        }
        text = json.dumps(report, indent=2)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
    finally:
        for worker in workers:
            worker.close()


if __name__ == "__main__":
    main()
