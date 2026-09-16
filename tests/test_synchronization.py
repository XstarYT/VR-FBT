from dataclasses import replace
import unittest
from Lib.Synchronization import FrameSynchronizer, MediaTimeline
from Lib.Tracking import CameraObservation, PoseResult
from Lib.Config import CameraSetup, Profile, validate_profile, camera_setup_for_corner, ConfigurationError


def observation(source, stamp):
    # A moving point gives a measurable error if different times are combined.
    return CameraObservation(source, PoseResult(True, 1, [[stamp * 3, 0, 0, 1]], []), (640, 480), stamp)


class SynchronizationTests(unittest.TestCase):
    def test_buffer_adapts_to_completion_jitter_without_rewinding_or_exceeding_budget(self):
        sync = FrameSynchronizer(('a', 'b'))
        previous = -1
        pairs = 0
        events = []
        for index in range(90):
            stamp = index / 15
            events.extend([(stamp + .01, observation('a', stamp)),
                           (stamp + (.13 if index % 3 else .08), observation('b', stamp))])
        events.sort(key=lambda pair: pair[0])
        cursor = 0
        for tick in range(180):
            now = tick / 30
            while cursor < len(events) and events[cursor][0] <= now:
                sync.add(events[cursor][1], now)
                cursor += 1
            selected = sync.select(now)
            self.assertGreaterEqual(sync.last_target, previous)
            previous = sync.last_target
            self.assertLessEqual(sync.alignment_delay, .2)
            if len(selected) == 2:
                pairs += 1
                self.assertAlmostEqual(selected[0].captured_at, selected[1].captured_at)
        self.assertGreater(sync.alignment_delay, .13)
        self.assertGreater(pairs, 150)

    def test_delayed_inference_uses_same_instant_from_fast_camera_history(self):
        sync = FrameSynchronizer(('fast', 'slow'))
        for index in range(31):
            stamp = index / 100
            sync.add(observation('fast', stamp))
            if stamp <= 0.22:
                sync.add(observation('slow', stamp))
        result = sync.select(0.301)
        self.assertEqual(len(result), 2)
        self.assertAlmostEqual(result[0].captured_at, .2)
        self.assertAlmostEqual(result[1].captured_at, .2)
        self.assertEqual(result[0].pose.image_landmarks, result[1].pose.image_landmarks)

    def test_delayed_view_is_excluded_instead_of_triangulating_different_motion(self):
        sync = FrameSynchronizer(('fast', 'late', 'healthy'))
        for source, stamp in [('fast', .20), ('healthy', .18), ('late', .10)]:
            sync.add(observation(source, stamp))
        self.assertEqual({item.source_id for item in sync.select(.301)}, {'fast', 'healthy'})

    def test_stall_does_not_block_healthy_camera_and_old_results_expire(self):
        sync = FrameSynchronizer(('fast', 'stalled'))
        sync.add(observation('stalled', 0))
        sync.add(observation('fast', .2))
        self.assertEqual([item.source_id for item in sync.select(.301)], ['fast'])
        self.assertEqual(sync.select(.6), [])

    def test_different_rates_stay_within_skew_and_never_rewind(self):
        sync = FrameSynchronizer(('a', 'b'))
        previous = {}
        pairs = 0
        for index in range(200):
            now = index / 100
            if index % 3 == 0:
                sync.add(observation('a', now))
            if index % 7 == 0:
                sync.add(observation('b', now - .06))
            results = sync.select(now)
            for item in results:
                self.assertGreaterEqual(item.captured_at, previous.get(item.source_id, -1))
                previous[item.source_id] = item.captured_at
            if len(results) == 2:
                pairs += 1
                self.assertLessEqual(abs(results[0].captured_at - results[1].captured_at), .033)
        self.assertGreater(pairs, 20)

    def test_single_camera_has_no_alignment_delay_and_history_is_bounded(self):
        sync = FrameSynchronizer(('a',))
        for index in range(1000):
            sync.add(observation('a', index / 1000))
        self.assertLessEqual(len(sync.histories['a']), 64)
        self.assertEqual(sync.select(1)[0].captured_at, .999)
        self.assertFalse(sync.add(observation('a', .1)))
        self.assertFalse(sync.add(observation('a', float('nan'))))
        self.assertEqual(sync.select(1, {'a'}), [])

    def test_media_timeline_exposes_buffering_instead_of_relabelling_as_fresh(self):
        timeline = MediaTimeline()
        self.assertEqual(timeline.timestamp(0, 100), 100)
        self.assertAlmostEqual(timeline.timestamp(.033, 100.233), 100.033)
        self.assertIsNone(timeline.timestamp(.033, 100.3))
        self.assertIsNone(timeline.timestamp(.01, 100.4))
        self.assertAlmostEqual(timeline.timestamp(.4, 100.41), 100.4)

    def test_residual_latency_is_validated_and_preserved_by_corner_placement(self):
        from Lib.Config import CAMERA_CORNER_SIGNS
        setup = CameraSetup('local:0', (0, 1.4, -2), (0, 0, 0), latency_ms=75)
        placed = camera_setup_for_corner(setup, next(iter(CAMERA_CORNER_SIGNS)), 1.4, (4, 2.7, 4))
        self.assertEqual(placed.latency_ms, 75)
        for value in (-1, 201, float('nan'), float('inf')):
            with self.assertRaises(ConfigurationError):
                validate_profile(Profile(camera_setups=(replace(setup, latency_ms=value),)))
