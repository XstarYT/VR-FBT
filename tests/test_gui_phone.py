import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from Lib.GUI import VRFBTApp
from Lib.Config import CameraSetup
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

    def test_form_selects_up_to_three_camera_sources(self):
        self.app.phone_hub.registry.connect('first', 'Front', 'one')
        self.app.phone_hub.registry.connect('second', 'Side', 'two')
        self.app._set_camera_options([])
        labels = {source: label for label, source in self.app.camera_sources.items()}
        self.app.camera_choices[0].set(labels['phone:first'])
        self.app.camera_choices[1].set(labels['phone:second'])
        self.app.camera_choices[2].set('Off')
        _, profile = self.app._configuration_from_form()
        self.assertEqual(profile.camera_sources, ('phone:first', 'phone:second'))
        self.assertEqual(profile.tracking_mode, 'MULTI')

    def test_form_preserves_manual_room_and_camera_layout(self):
        self.app.phone_hub.registry.connect('first', 'Front', 'one')
        self.app.phone_hub.registry.connect('second', 'Side', 'two')
        self.app._set_camera_options([])
        labels = {source: label for label, source in self.app.camera_sources.items()}
        self.app.camera_choices[0].set(labels['phone:first'])
        self.app.camera_choices[1].set(labels['phone:second'])
        self.app.manual_camera_setup.set(True)
        self.app.room_width.set('5.0'); self.app.room_height.set('3.0'); self.app.room_depth.set('6.0')
        self.app.camera_setup_values = {
            'phone:first': CameraSetup('phone:first', (0.0, 1.5, -2.5), (0.0, -5.0, 0.0), 65.0),
            'phone:second': CameraSetup('phone:second', (2.5, 1.5, 0.0), (-90.0, -5.0, 0.0), 70.0),
        }
        _, profile = self.app._configuration_from_form()
        self.assertTrue(profile.manual_camera_setup)
        self.assertEqual(profile.room_size_m, (5.0, 3.0, 6.0))
        self.assertEqual(profile.camera_setups, tuple(self.app.camera_setup_values.values()))

    def test_camera_layout_dialog_opens_for_current_sources(self):
        before = set(self.app.winfo_children())
        self.app._show_camera_setup()
        self.app.update_idletasks()
        dialogs = [child for child in self.app.winfo_children() if child not in before and child.winfo_class() == 'Toplevel']
        self.assertEqual(len(dialogs), 1)
        self.assertEqual(dialogs[0].title(), 'Camera and room layout')
        dialogs[0].destroy()

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
