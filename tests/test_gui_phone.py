import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from Lib.GUI import VRFBTApp
from Lib.RemoteCam import ensure_local_certificates


class PhoneGuiTests(unittest.TestCase):
    def setUp(self):
        self.scan = patch.object(VRFBTApp, '_refresh_cameras')
        self.scan.start()
        self.app = VRFBTApp()
        self.app.update_idletasks()
        self.app.withdraw()

    def tearDown(self):
        self.app._on_close()
        self.scan.stop()

    def test_local_connection_shows_one_camera_qr_and_stops_cleanly(self):
        with tempfile.TemporaryDirectory() as directory:
            certificate_paths = ensure_local_certificates(Path(directory))
            with patch('Lib.GUI.ensure_local_certificates', return_value=certificate_paths):
                self.app._start_phone_server()
                deadline = time.monotonic() + 5
                while not self.app.primary_phone_url and time.monotonic() < deadline:
                    self.app.update()
                    time.sleep(0.02)
                self.assertTrue(self.app.primary_phone_url.startswith('https://'))
                self.assertIn('/#token=', self.app.primary_phone_url)
                self.assertEqual(self.app.phone_hub.host, '0.0.0.0')
                self.assertTrue(self.app.phone_hub.running)
                self.assertTrue(self.app._phone_dialog.winfo_exists())
                self.assertEqual(len(self.app._phone_dialog.qr_images), 1)
                self.assertFalse(hasattr(self.app, 'phone_mode'))
                self.assertFalse(hasattr(self.app, 'certificate_phone_url'))
                self.app._stop_phone_server()
                self.assertFalse(self.app.phone_hub.running)
                self.assertEqual(self.app.primary_phone_url, '')

    def test_same_phone_names_remain_selectable(self):
        self.app.phone_hub.registry.connect('first', 'Rear camera', 'one')
        self.app.phone_hub.registry.connect('second', 'Rear camera', 'two')
        self.app._set_camera_options([])
        self.assertIn('phone:first', self.app.camera_sources.values())
        self.assertIn('phone:second', self.app.camera_sources.values())

    def test_vrchat_height_is_read_from_form(self):
        self.app.user_height.set('1.83')
        _, profile = self.app._configuration_from_form()
        self.assertEqual(profile.user_height_m, 1.83)

    def test_stable_tracker_set_is_default(self):
        _, profile = self.app._configuration_from_form()
        self.assertEqual(profile.vrchat_tracker_set, "stable")

    def test_realign_button_calls_controller(self):
        with patch.object(self.app.controller, 'realign_vrchat') as realign:
            self.app._realign_vrchat()
        realign.assert_called_once_with()

    def test_form_controls_use_high_contrast_text(self):
        style = self.app.tk.call
        entry_background = style('ttk::style', 'lookup', 'TEntry', '-fieldbackground')
        entry_foreground = style('ttk::style', 'lookup', 'TEntry', '-foreground')
        combo_background = style('ttk::style', 'lookup', 'TCombobox', '-fieldbackground', 'readonly')
        combo_foreground = style('ttk::style', 'lookup', 'TCombobox', '-foreground', 'readonly')
        self.assertEqual(entry_background.lower(), '#263653')
        self.assertEqual(entry_foreground.lower(), '#ffffff')
        self.assertEqual(combo_background.lower(), '#263653')
        self.assertEqual(combo_foreground.lower(), '#ffffff')

    def test_vrchat_osc_button_sends_test_pulse(self):
        messages = []

        class FakeServer:
            target = ("127.0.0.1", 9000)
            def __init__(self, host, port):
                self.requested = (host, port)
            def Send(self, message):
                messages.append(message)
            def close(self):
                pass

        with patch('Lib.OSCKit.Server', FakeServer):
            self.app.osc_host.set('localhost')
            self.app.osc_port.set('9000')
            self.app._test_vrchat_osc()
        self.assertEqual(messages, [
            ('/avatar/parameters/VRFBT_ConnectionTest', 1.0),
            ('/avatar/parameters/VRFBT_ConnectionTest', 0.0),
        ])


if __name__ == '__main__':
    unittest.main()
