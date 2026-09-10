import threading
import unittest

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer

from Lib import OSCKit
from Lib.Engine import TrackingController
from Lib.VRChat import TrackerPose, VRChatFrame


class VRChatUDPIntegrationTests(unittest.TestCase):
    def test_complete_frame_reaches_a_real_udp_osc_receiver(self):
        received = []
        complete = threading.Event()

        def handler(address, *values):
            received.append((address, values))
            if len(received) >= 18:
                complete.set()

        dispatcher = Dispatcher()
        dispatcher.set_default_handler(handler)
        receiver = ThreadingOSCUDPServer(("127.0.0.1", 0), dispatcher)
        receiver_thread = threading.Thread(target=receiver.serve_forever, daemon=True)
        receiver_thread.start()
        client = OSCKit.Server("127.0.0.1", receiver.server_address[1])
        try:
            trackers = {
                str(index): TrackerPose(
                    (index / 10, index / 20, index / 30),
                    (index * 1.0, index * 2.0, index * 3.0),
                    0.95,
                )
                for index in range(1, 9)
            }
            frame = VRChatFrame(trackers, (0.1, 1.7, 0.2), (0.0, 15.0, 0.0), 1.0, 0.95)
            TrackingController._send_vrchat_frame(frame, client, OSCKit, align_head=True)
            self.assertTrue(complete.wait(2), f"Only received {len(received)} OSC messages")

            by_path = {path: values for path, values in received}
            expected_paths = {
                f"/tracking/trackers/{index}/{kind}"
                for index in range(1, 9)
                for kind in ("position", "rotation")
            }
            expected_paths.update({
                "/tracking/trackers/head/position",
                "/tracking/trackers/head/rotation",
            })
            self.assertEqual(set(by_path), expected_paths)
            self.assertEqual(len(by_path["/tracking/trackers/2/position"]), 3)
            self.assertAlmostEqual(by_path["/tracking/trackers/2/position"][0], 0.2, places=5)
            self.assertEqual(tuple(round(value) for value in by_path["/tracking/trackers/8/rotation"]), (8, 16, 24))
        finally:
            client.close()
            receiver.shutdown()
            receiver.server_close()
            receiver_thread.join(2)


if __name__ == "__main__":
    unittest.main()
