import gc
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from Lib.GUI import VRFBTApp
from Lib.Config import CameraSetup, Profile
from Lib.RemoteCam import LocalCamera, ensure_local_certificates
from tests.gui_process_case import FreshProcessGuiTest


class PhoneGuiTests(FreshProcessGuiTest):
    def test_camera_health_table_uses_anonymous_rows_and_handles_missing_samples(self):
        from Lib.Metrics import SessionMetrics
        metrics = SessionMetrics(("phone:private-id", "local:0"))
        self.app._show_health(metrics.snapshot())
        rows = [self.app.health_table.item(item)["values"] for item in self.app.health_table.get_children()]
        self.assertEqual([row[0] for row in rows], ["camera-1", "camera-2"])
        self.assertEqual(rows[0][1], "waiting")
        self.assertEqual(rows[0][3], "—")
        metrics.fail("local:0")
        self.app._show_health(metrics.snapshot())
        rows = [self.app.health_table.item(item)["values"] for item in self.app.health_table.get_children()]
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[1][1], "failed")

    def test_activity_health_table_has_readable_text(self):
        style = self.app.tk.call
        background = style("ttk::style", "lookup", "Treeview", "-background")
        foreground = style("ttk::style", "lookup", "Treeview", "-foreground")
        self.assertEqual(background.lower(), "#182238")
        self.assertEqual(foreground.lower(), "#ffffff")

    def test_save_as_preserves_old_rig_and_selects_new_profile(self):
        from Lib import Config
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.multiple(Config, PROFILES_DIR=root / "profiles", SETTINGS_PATH=root / "settings.json"):
                settings, profile = self.app._configuration_from_form()
                Config.save_configuration(settings, profile)
                original = (Config.PROFILES_DIR / f"{profile.name}.toml").read_bytes()
                self.app.fps.set("24")
                with patch("Lib.GUI.simpledialog.askstring", return_value="Alternate rig"):
                    self.app._save_as()
                self.assertEqual(self.app.profile_name.get(), "Alternate rig")
                self.assertEqual(self.app.loaded_settings.fps, 24)
                self.assertEqual(self.app.loaded_profile.name, "Alternate rig")
                self.assertEqual(Config.load_settings(Config.SETTINGS_PATH).default_profile, "Alternate rig")
                self.assertEqual((Config.PROFILES_DIR / f"{profile.name}.toml").read_bytes(), original)

    def test_save_updates_the_in_memory_saved_profile(self):
        with patch("Lib.GUI.save_configuration") as save:
            self.app.fps.set("45")
            self.app.user_height.set("1.89")
            self.assertTrue(self.app._save(quiet=True))
        save.assert_called_once()
        self.assertEqual(self.app.loaded_settings.fps, 45)
        self.assertEqual(self.app.loaded_profile.user_height_m, 1.89)

    def test_small_window_keeps_tracking_controls_visible_and_fields_reachable(self):
        self.app.deiconify()

        def descendants(widget):
            for child in widget.winfo_children():
                yield child
                yield from descendants(child)

        osc_port = next(widget for widget in descendants(self.app.setup_scroll.content)
                        if widget.winfo_class() == "TEntry" and str(widget.cget("textvariable")) == str(self.app.osc_port))
        for size in ("940x640", "1120x760"):
            with self.subTest(size=size):
                self.app.geometry(size)
                # Native mapping requires event processing, not only geometry
                # idle callbacks; checking 1x1 unmapped widgets is misleading.
                for _ in range(3):
                    self.app.update()
                    time.sleep(0.02)
                for button in (self.app.start_button, self.app.stop_button, self.app.align_button, self.app.save_as_button):
                    self.assertTrue(button.winfo_ismapped())
                    self.assertGreater(button.winfo_width(), 20)
                    self.assertLessEqual(button.winfo_rooty() + button.winfo_height(), self.app.winfo_rooty() + self.app.winfo_height())
                    self.assertGreaterEqual(button.winfo_rootx(), self.app.winfo_rootx())
                    self.assertLessEqual(button.winfo_rootx() + button.winfo_width(), self.app.winfo_rootx() + self.app.winfo_width())
                for field in (self.app.camera_combos[0], osc_port):
                    field.event_generate("<FocusIn>")
                    self.app.update()
                    canvas = self.app.setup_scroll.canvas
                    self.assertGreaterEqual(field.winfo_rooty(), canvas.winfo_rooty())
                    self.assertLessEqual(field.winfo_rooty() + field.winfo_height(), canvas.winfo_rooty() + canvas.winfo_height())
                    self.assertGreaterEqual(field.winfo_rootx(), canvas.winfo_rootx())
                    self.assertLessEqual(field.winfo_rootx() + field.winfo_width(), canvas.winfo_rootx() + canvas.winfo_width())

    def test_support_export_cancel_does_not_start_background_work(self):
        with patch("Lib.GUI.filedialog.asksaveasfilename", return_value=""):
            self.app._export_support()
        self.assertFalse(hasattr(self.app, "_export_thread"))

    def test_support_export_uses_visible_settings_and_reports_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = str(Path(directory) / "support.zip")
            self.app.fps.set("24")
            with patch("Lib.GUI.filedialog.asksaveasfilename", return_value=destination), patch("Lib.Support.export_support_bundle") as export, patch("Lib.GUI.messagebox.showinfo") as notice:
                self.app._export_support()
                self.app._export_thread.join(3)
                self.app._drain_events()
                export.assert_called_once()
                self.assertEqual(export.call_args.args[1].fps, 24)
                self.assertEqual(export.call_args.args[0], destination)
                self.assertEqual(str(self.app.export_button["state"]), "normal")
                notice.assert_called_once()

    def setUp(self):
        self.scan = patch.object(VRFBTApp, '_refresh_cameras')
        self.scan.start()
        self.app = VRFBTApp()
        self.app.update_idletasks()
        self.app.withdraw()

    def tearDown(self):
        self.app._on_close()
        # Engine callbacks close over the app; collect that cycle while Tcl is
        # still being exercised on the test's main thread rather than during a
        # later asyncio server test.
        self.app = None
        gc.collect()
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

    def test_identically_named_local_cameras_have_stable_distinct_labels(self):
        self.app._set_camera_options([
            LocalCamera(0, "USB Camera"), LocalCamera(1, "USB Camera"),
        ])
        labels = {source: label for label, source in self.app.camera_sources.items()}
        self.assertIn("local:0", labels)
        self.assertIn("local:1", labels)
        self.assertNotEqual(labels["local:0"], labels["local:1"])
        self.assertIn("local:0", labels["local:0"])
        self.assertIn("local:1", labels["local:1"])

    def test_profile_sources_are_rebuilt_and_selected_even_when_offline(self):
        sources = ("phone:saved-offline", "local:8")
        self.app.loaded_profile = Profile(
            camera_source=sources[0], camera_sources=sources, tracking_mode="MULTI",
        )
        self.app._set_camera_options([], sources)
        selected = tuple(
            self.app.camera_sources.get(choice.get())
            for choice in self.app.camera_choices[:2]
        )
        self.assertEqual(selected, sources)

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

    def test_start_blocks_when_selected_phone_is_offline(self):
        self.app.loaded_profile = Profile(camera_source="phone:missing", camera_sources=("phone:missing",))
        self.app._set_camera_options([], ("phone:missing",))
        with patch.object(self.app, "_save", return_value=True), \
             patch.object(self.app.controller, "start") as start, \
             patch("Lib.GUI.messagebox.showerror") as error:
            self.app._start()
        start.assert_not_called()
        self.assertIn("phone:missing", error.call_args.args[1])

    def test_saved_local_placeholder_is_not_counted_as_available_override(self):
        selected = ("phone:missing", "local:8")
        self.app.loaded_profile = Profile(camera_source=selected[0], camera_sources=selected, tracking_mode="MULTI")
        self.app._set_camera_options([], selected)
        self.assertEqual(self.app.discovered_local_sources, set())
        with patch.object(self.app, "_save", return_value=True), \
             patch.object(self.app.controller, "start") as start, \
             patch("Lib.GUI.messagebox.showerror") as error, \
             patch("Lib.GUI.messagebox.askyesno") as offer:
            self.app._start()
        start.assert_not_called()
        offer.assert_not_called()
        error.assert_called_once()

    def test_start_can_use_connected_subset_without_changing_saved_selection(self):
        selected = ("phone:online", "phone:missing")
        self.app.phone_hub.registry.connect("online", "Front", "local")
        self.app.loaded_profile = Profile(camera_source=selected[0], camera_sources=selected, tracking_mode="MULTI")
        self.app._set_camera_options([], selected)
        with patch.object(self.app, "_save", return_value=True), \
             patch.object(self.app.controller, "start") as start, \
             patch("Lib.GUI.messagebox.askyesno", return_value=True):
            self.app._start()
        session = start.call_args.args[1]
        self.assertEqual(session.camera_sources, ("phone:online",))
        self.assertEqual(session.camera_source, "phone:online")
        self.assertEqual(session.tracking_mode, "SINGLE")
        self.assertEqual(self.app.loaded_profile.camera_sources, selected)

    def test_setup_wizard_saves_stable_automatic_profile_and_completion_flag(self):
        from Lib import Config
        self.app._set_camera_options([LocalCamera(0, "Webcam")])
        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(Config, PROFILES_DIR=Path(directory) / "profiles", SETTINGS_PATH=Path(directory) / "settings.json"), \
             patch("Lib.Wizard.run_diagnostics", return_value=True):
            self.app._show_wizard()
            wizard = self.app._setup_wizard
            deadline = time.monotonic() + 2
            while not wizard.diagnostics_ok and time.monotonic() < deadline:
                self.app.update()
                time.sleep(0.02)
            self.assertTrue(wizard.diagnostics_ok)
            wizard.next()  # VRChat OSC
            wizard.osc_tested = True
            wizard.next()  # Connect camera
            wizard._choose_local()
            wizard.next()  # Choose camera
            wizard.next()  # User height
            wizard.height.set("1.83")
            wizard.next()  # Trackers
            wizard.next()  # Summary
            wizard.next()  # Save
            self.assertTrue(Config.load_settings(Config.SETTINGS_PATH).first_run_completed)
            saved = Config.load_profile("Default")
            self.assertFalse(saved.manual_camera_setup)
            self.assertEqual(saved.user_height_m, 1.83)
            self.assertEqual(saved.vrchat_tracker_set, "stable")
            self.assertEqual(saved.camera_sources, ("local:0",))

    def test_setup_wizard_blocks_continue_after_failed_diagnostics(self):
        with patch("Lib.Wizard.run_diagnostics", return_value=False):
            self.app._show_wizard()
            wizard = self.app._setup_wizard
            deadline = time.monotonic() + 2
            while wizard.results.empty() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.app.update()
            with patch("Lib.Wizard.messagebox.showerror") as error:
                wizard.next()
            error.assert_called_once()
            self.assertEqual(wizard.step, 0)
            wizard.close()

    def test_start_closes_capture_preview_before_opening_tracking_camera(self):
        self.app._set_camera_options([LocalCamera(0, "Webcam")], desired_sources=("local:0",))
        with patch("Lib.Preview.CapturePreview") as preview_class, \
             patch.object(self.app, "_save", return_value=True), \
             patch.object(self.app.controller, "start") as start:
            preview = preview_class.return_value
            preview.stop.return_value = True
            self.app._toggle_preview()
            preview.start.assert_called_once()
            self.app._start()
            preview.stop.assert_called_once()
            start.assert_called_once()
        self.assertIsNone(self.app._preview)

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

    def test_lens_import_is_applied_only_when_layout_is_applied(self):
        import json
        def widgets(parent):
            for child in parent.winfo_children():
                yield child
                yield from widgets(child)
        intrinsics=(960.,720.,820.,825.,475.,354.)
        distortion=(-.18,.04,.001,-.002,0.)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'lens.json'
            path.write_text(json.dumps({'schema':'vr-fbt-lens-v1','intrinsics':intrinsics,'distortion':distortion}))
            original=dict(self.app.camera_setup_values)
            self.app._show_camera_setup()
            dialog=next(child for child in self.app.winfo_children() if child.winfo_class()=='Toplevel')
            with patch('Lib.GUI.filedialog.askopenfilename',return_value=str(path)):
                next(w for w in widgets(dialog) if w.winfo_class()=='TButton' and w.cget('text')=='Import lens…').invoke()
            self.assertEqual(self.app.camera_setup_values,original)
            next(w for w in widgets(dialog) if w.winfo_class()=='TButton' and w.cget('text')=='Apply layout').invoke()
            _,profile=self.app._configuration_from_form()
            self.assertEqual(profile.camera_setups[0].lens_intrinsics,intrinsics)
            self.assertEqual(profile.camera_setups[0].lens_distortion,distortion)

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
