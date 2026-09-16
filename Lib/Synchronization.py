"""Bounded camera timelines; never triangulate widely separated poses."""

from collections import deque
import math


MAX_FRAME_AGE = 0.250
MAX_VIEW_SKEW = 0.033
ALIGNMENT_DELAY = 0.100


class FrameSynchronizer:
    """Play multiple sources at a common adaptive 100-200 ms horizon without waiting on I/O.

    Pose histories, not images, are retained. A missing/late source is excluded.
    A single configured camera needs no alignment delay. Timestamps must be in
    the PC monotonic clock, with any known residual device delay subtracted.
    """

    def __init__(self, sources):
        self.histories = {source: deque(maxlen=64) for source in sources}
        self.last_target = float("-inf")
        self.delays = {source: deque(maxlen=32) for source in self.histories}
        self.alignment_delay = ALIGNMENT_DELAY

    def add(self, observation, completed_at=None):
        stamp = observation.captured_at
        history = self.histories.get(observation.source_id)
        if history is None or stamp is None or not math.isfinite(stamp):
            return False
        if history and stamp <= history[-1].captured_at:
            return False
        history.append(observation)
        if completed_at is not None:
            delay = completed_at - stamp
            if math.isfinite(delay) and 0 <= delay <= 0.200:
                self.delays[observation.source_id].append(delay)
        return True

    def select(self, now, excluded=()):
        # A fixed 100 ms horizon can repeatedly exclude an otherwise usable
        # camera whose inference/transport jitter occasionally exceeds it.
        # Adapt within a hard 200 ms budget; never wait indefinitely for a source.
        observed = [delay for source, delays in self.delays.items() if source not in excluded for delay in delays]
        self.alignment_delay = min(.200, max(ALIGNMENT_DELAY, max(observed, default=0) + .005))
        target = now - (self.alignment_delay if len(self.histories) > 1 else 0)
        target = max(target, self.last_target)
        self.last_target = target
        candidates = []
        for source, history in self.histories.items():
            while history and now - history[0].captured_at > MAX_FRAME_AGE:
                history.popleft()
            if source in excluded:
                history.clear()
                continue
            candidate = next((item for item in reversed(history) if item.captured_at <= target), None)
            if candidate is not None:
                candidates.append(candidate)
        if not candidates:
            return []
        newest = max(item.captured_at for item in candidates)
        return [item for item in candidates if newest - item.captured_at <= MAX_VIEW_SKEW]


class MediaTimeline:
    """Map a relative media clock to earliest observed PC arrival.

    This exposes *additional* buffering/jitter, not constant sensor/network
    delay. A new connection creates a new timeline; a backwards media clock
    requires reconnecting rather than disguising an old frame as a new one.
    """

    def __init__(self):
        self.offset = None
        self.last_media = None

    def timestamp(self, media_time, arrived_at):
        if media_time is None or not math.isfinite(media_time):
            return arrived_at
        if self.last_media is not None and media_time <= self.last_media:
            return None  # duplicate/out-of-order video must not appear fresh
        observed_offset = arrived_at - media_time
        self.offset = observed_offset if self.offset is None else min(self.offset, observed_offset)
        self.last_media = media_time
        return media_time + self.offset
