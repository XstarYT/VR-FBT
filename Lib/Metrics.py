"""Bounded numeric session telemetry; never retains camera images or landmarks."""

from collections import deque
import math
import threading
import time


SAMPLE_LIMIT = 512
RATE_WINDOW = 2.0


def percentiles(samples):
    values = sorted(samples)
    if not values:
        return {"count": 0, "p50_ms": None, "p95_ms": None, "p99_ms": None}

    def at(fraction):
        index = (len(values) - 1) * fraction
        lower = int(index)
        upper = min(lower + 1, len(values) - 1)
        return round(values[lower] + (values[upper] - values[lower]) * (index - lower), 3)

    return {"count": len(values), "p50_ms": at(0.5), "p95_ms": at(0.95), "p99_ms": at(0.99)}


class SessionMetrics:
    def __init__(self, sources=(), clock=time.monotonic):
        self.clock = clock
        self.started = clock()
        self.ended = None
        self._lock = threading.Lock()
        self._updates = deque(maxlen=2048)
        self._stages = {name: deque(maxlen=SAMPLE_LIMIT) for name in ("fusion", "osc_send")}
        self._packets = 0
        self._sources = {source: {
            "alias": f"camera-{index + 1}", "kind": "phone" if source.startswith("phone:") else "local",
            "captures": 0, "selected_sequence": 0, "capture_skipped": 0,
            "stale_rejected": 0, "failures": 0, "pose_detected": False,
            "last_capture": None, "last_result_capture": None,
            "synchronized": None,
            "times": deque(maxlen=2048), "inference": deque(maxlen=SAMPLE_LIMIT),
            "capture_to_result": deque(maxlen=SAMPLE_LIMIT),
        } for index, source in enumerate(sources)}

    def capture(self, source, captured_at):
        with self._lock:
            row = self._sources.get(source)
            if row is not None and math.isfinite(captured_at):
                row["captures"] += 1
                row["last_capture"] = captured_at
                row["times"].append(captured_at)

    def select(self, source, sequence):
        with self._lock:
            row = self._sources.get(source)
            if row is not None and sequence > row["selected_sequence"]:
                row["capture_skipped"] += max(0, sequence - row["selected_sequence"] - 1)
                row["selected_sequence"] = sequence

    def reject_stale(self, source):
        with self._lock:
            if source in self._sources:
                self._sources[source]["stale_rejected"] += 1

    def fail(self, source):
        with self._lock:
            if source in self._sources:
                self._sources[source]["failures"] += 1

    def inference(self, source, duration_ms, captured_at, completed_at, detected):
        with self._lock:
            row = self._sources.get(source)
            if row is not None:
                if math.isfinite(duration_ms) and duration_ms >= 0:
                    row["inference"].append(duration_ms)
                age = (completed_at - captured_at) * 1000
                if math.isfinite(age) and age >= 0:
                    row["capture_to_result"].append(age)
                row["last_result_capture"] = captured_at
                row["pose_detected"] = bool(detected)

    def stage(self, name, duration_ms, packets=0):
        with self._lock:
            if name in self._stages and math.isfinite(duration_ms) and duration_ms >= 0:
                self._stages[name].append(duration_ms)
            self._packets += max(0, int(packets))

    def synchronization(self, observations):
        selected = {item.source_id for item in observations}
        with self._lock:
            for source, row in self._sources.items():
                row["synchronized"] = source in selected

    def update(self):
        with self._lock:
            now = self.clock()
            self._updates.append(now)
            window = max(0.001, min(RATE_WINDOW, now - self.started))
            return round(sum(now - window < stamp <= now for stamp in self._updates) / window, 1)

    def finish(self):
        with self._lock:
            self.ended = self.clock()

    def snapshot(self):
        with self._lock:
            now = self.clock() if self.ended is None else self.ended
            window = max(0.001, min(RATE_WINDOW, now - self.started))

            def rate(times):
                return round(sum(now - window < stamp <= now for stamp in times) / window, 1)

            rows = []
            for row in self._sources.values():
                age = None if row["last_capture"] is None else max(0, now - row["last_capture"])
                result_age = None if row["last_result_capture"] is None else max(0, now - row["last_result_capture"])
                status = ("failed" if row["failures"] else "stopped" if self.ended is not None else
                          "waiting" if age is None else "stale" if age > 0.5 else
                          "waiting for inference" if result_age is None else "lagging" if result_age > 0.5 else
                          "out of sync" if row["synchronized"] is False else
                          "pose visible" if row["pose_detected"] else "no pose")
                rows.append({key: row[key] for key in ("alias", "kind", "captures", "capture_skipped", "stale_rejected", "failures")})
                rows[-1].update(status=status, capture_fps=rate(row["times"]),
                    frame_age_ms=None if age is None else round(age * 1000, 1),
                    inference=percentiles(row["inference"]), capture_to_result=percentiles(row["capture_to_result"]))
            return {"running": self.ended is None, "duration_seconds": round(max(0, now - self.started), 3),
                    "rate_window_seconds": RATE_WINDOW, "latency_sample_limit": SAMPLE_LIMIT,
                    "update_fps": rate(self._updates), "osc_packets_sent": self._packets,
                    "stages": {name: percentiles(samples) for name, samples in self._stages.items()},
                    "sources": rows}
