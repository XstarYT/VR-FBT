"""Conservative epipolar consistency guard; does not estimate camera movement."""
from itertools import combinations
import numpy as np

JOINTS = (11,12,13,14,15,16,23,24,25,26,27,28)


def pair_error(first, second, projections, positions):
    """Median symmetric point-to-epipolar-line distance in 960-wide pixels."""
    p, q = projections[first.source_id], projections[second.source_id]
    center = np.append(positions[first.source_id], 1.)
    e = q @ center
    cross = np.array([[0,-e[2],e[1]],[e[2],0,-e[0]],[-e[1],e[0],0]])
    f = cross @ q @ np.linalg.pinv(p)
    if np.linalg.norm(f) < 1e-10:
        return float('inf')  # coincident camera centers cannot establish depth
    errors = []
    for joint in JOINTS:
        a,b = first.pose.image_landmarks[joint],second.pose.image_landmarks[joint]
        if min(a[3],b[3]) < .65:
            continue
        x = np.array([a[0]*(first.frame_size[0]-1),a[1]*(first.frame_size[1]-1),1.])
        y = np.array([b[0]*(second.frame_size[0]-1),b[1]*(second.frame_size[1]-1),1.])
        lx,ly = f.T@y,f@x
        nx,ny = np.linalg.norm(lx[:2]),np.linalg.norm(ly[:2])
        if min(nx,ny) < 1e-10:
            continue
        distance = abs(y@f@x)
        errors.append(.5*distance*(960/first.frame_size[0]/nx+960/second.frame_size[0]/ny))
    return float(np.median(errors)) if len(errors) >= 6 else None


class GeometryHealth:
    def __init__(self):
        self.quarantined = set()
        self._evidence = {}
        self._signatures = {}
        self._latched_all = False

    def filter(self, observations, projections, positions, now):
        if self._latched_all:
            return [], 'Camera views disagree with calibration; check layout/timing and Recalibrate', True
        available = [o for o in observations if o.source_id not in self.quarantined]
        if len(available) < 2 and self.quarantined:
            return [], 'Not enough trusted cameras; restore camera layout and Recalibrate', True
        if len(available) < 2:
            return available, '', False
        errors = {tuple(sorted((a.source_id,b.source_id))): pair_error(a,b,projections,positions)
                  for a,b in combinations(available,2)}
        bad = [pair for pair,error in errors.items() if error is not None and error > 20.]
        good = [pair for pair,error in errors.items() if error is not None and error <= 10.]
        suspect = None
        if len(available) == 3 and len(good) == 1 and len(bad) == 2:
            suspect = next(o.source_id for o in available if o.source_id not in good[0])
        ambiguous = len(available) == 2 and bool(bad) or len(bad) >= 2 and suspect is None
        key = suspect or ('all' if ambiguous else None)
        if key is None:
            self._evidence.clear()
            warning = 'Excluded camera(s): '+', '.join(sorted(self.quarantined))+'; check layout and Recalibrate' if self.quarantined else ''
            return available, warning, False
        # Reusing the same synchronized capture must never add evidence.
        signature = tuple((o.source_id,o.captured_at) for o in available)
        if all(o.captured_at is not None for o in available) and self._signatures.get(key) != signature:
            start,count,last = self._evidence.get(key,(now,0,now))
            if now-last > .5:
                start,count = now,0
            self._evidence = {key:(start,count+1,now)}
            self._signatures[key] = signature
            if count+1 >= 8 and now-start >= .75:
                if suspect:
                    self.quarantined.add(suspect)
                else:
                    self._latched_all = True
        if suspect:
            return [o for o in available if o.source_id != suspect], 'Camera '+suspect+' disagrees; excluded while checking calibration', False
        return [], 'Camera views disagree; output paused. Check camera layout, lens calibration and timestamps', True
