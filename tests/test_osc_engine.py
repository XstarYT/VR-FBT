import unittest
import socket

from Lib import OSCKit
from Lib.Engine import TrackingController
from Lib.VRChat import TrackerPose, VRChatFrame


class FakeServer:
    def __init__(self):
        self.messages = []

    def Send(self, message):
        self.messages.append(message)


class OscAndEngineTests(unittest.TestCase):
    def test_osc_parsing_is_literal_only(self):
        self.assertEqual(OSCKit.Phrase.str("/test|[1, 2, 3]"), ("/test", [1, 2, 3]))
        with self.assertRaises((ValueError, SyntaxError)):
            OSCKit.Phrase.str("/test|__import__('os').getcwd()")

    def test_localhost_is_forced_to_vrchat_ipv4_listener(self):
        server = OSCKit.Server("localhost", 9000)
        try:
            self.assertEqual(server.target, ("127.0.0.1", 9000))
            self.assertEqual(server.client._sock.family, socket.AF_INET)
        finally:
            server.close()

    def test_vrchat_frame_sends_eight_positions_and_rotations(self):
        trackers = {
            str(index): TrackerPose((index, index + 1, index + 2), (1, 2, 3), 0.9)
            for index in range(1, 9)
        }
        frame = VRChatFrame(trackers, (0, 1, 2), (3, 4, 5), 1.0, 0.9)
        server = FakeServer()
        TrackingController._send_vrchat_frame(frame, server, OSCKit)
        self.assertEqual(len(server.messages), 17)
        self.assertEqual(server.messages[0][0], "/tracking/trackers/1/position")
        self.assertEqual(len(server.messages[0][1]), 3)
        self.assertEqual(server.messages[1][0], "/tracking/trackers/1/rotation")
        self.assertEqual(server.messages[-1][0], "/tracking/trackers/head/position")

    def test_stable_set_sends_only_hip_and_feet(self):
        trackers = {
            str(index): TrackerPose((index, index + 1, index + 2), (1, 2, 3), 0.9)
            for index in range(1, 9)
        }
        server = FakeServer()
        frame = VRChatFrame(trackers, (0, 1, 2), (3, 4, 5), 1.0, 0.9)
        TrackingController._send_vrchat_frame(
            frame, server, OSCKit, follow_head=False, tracker_ids={"2", "7", "8"},
        )
        addresses = [message[0] for message in server.messages]
        self.assertEqual(len(addresses), 6)
        self.assertEqual({address.split("/")[3] for address in addresses}, {"2", "7", "8"})

    def test_invisible_tracker_and_optional_head_alignment(self):
        trackers = {
            "1": TrackerPose((0, 1, 2), (1, 2, 3), 0.1),
            "2": TrackerPose((3, 4, 5), (4, 5, 6), 0.9),
        }
        frame = VRChatFrame(trackers, (0, 1, 2), (3, 4, 5), 1.0, 0.9)
        server = FakeServer()
        TrackingController._send_vrchat_frame(frame, server, OSCKit, align_head=True)
        paths = {message[0] for message in server.messages}
        self.assertNotIn("/tracking/trackers/1/position", paths)
        self.assertNotIn("/tracking/trackers/1/rotation", paths)
        self.assertIn("/tracking/trackers/2/position", paths)
        self.assertIn("/tracking/trackers/head/position", paths)
        self.assertIn("/tracking/trackers/head/rotation", paths)

    def test_low_confidence_head_does_not_move_tracking_space(self):
        frame = VRChatFrame({"1": TrackerPose((0, 1, 2), (1, 2, 3), 0.9)}, (0, 1, 2), (3, 4, 5), 1.0, 0.2)
        server = FakeServer()
        TrackingController._send_vrchat_frame(frame, server, OSCKit, align_head=True)
        paths = {message[0] for message in server.messages}
        self.assertNotIn("/tracking/trackers/head/position", paths)
        self.assertNotIn("/tracking/trackers/head/rotation", paths)


if __name__ == "__main__":
    unittest.main()
