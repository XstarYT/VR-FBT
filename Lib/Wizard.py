"""First-run setup assistant for the maintained desktop interface."""

from dataclasses import replace
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from Lib.Config import run_diagnostics, save_configuration


class SetupWizard:
    STEPS = ("Diagnostics", "VRChat OSC", "Connect camera", "Choose camera", "User height", "Trackers", "Ready")

    def __init__(self, app):
        self.app = app
        self.step = 0
        self.diagnostics_ok = False
        self.osc_tested = False
        self.use_local = False
        self.results = queue.Queue()
        self.camera_choice = tk.StringVar(master=app)
        self.height = tk.StringVar(master=app, value=app.user_height.get())
        self.trackers = tk.StringVar(master=app, value="Stable — hip + feet (recommended)")
        self.dialog = tk.Toplevel(app)
        self.dialog.title("VR-FBT setup assistant")
        self.dialog.geometry("650x450")
        self.dialog.minsize(560, 400)
        self.dialog.transient(app)
        self.dialog.protocol("WM_DELETE_WINDOW", self.close)
        self.box = ttk.Frame(self.dialog, padding=22)
        self.box.pack(fill="both", expand=True)
        self.content = ttk.Frame(self.box)
        self.content.pack(fill="both", expand=True)
        navigation = ttk.Frame(self.box)
        navigation.pack(fill="x", pady=(14, 0))
        self.back_button = ttk.Button(navigation, text="Back", command=self.back)
        self.back_button.pack(side="left")
        self.next_button = ttk.Button(navigation, text="Continue", command=self.next)
        self.next_button.pack(side="right")
        self.render()
        self._run_diagnostics()
        self._poll()

    def close(self):
        if getattr(self.app, "_setup_wizard", None) is self:
            self.app._setup_wizard = None
        self.dialog.destroy()

    def _run_diagnostics(self):
        self.diagnostics_ok = False
        def work():
            try:
                ok = run_diagnostics(lambda name, passed, detail: self.app.events.put(
                    ("log", "PASS" if passed else "ERROR", f"{name}: {detail}")))
                self.results.put(("diagnostics", ok))
            except Exception as exc:
                self.results.put(("diagnostics_error", str(exc)))
        threading.Thread(target=work, name="wizard-diagnostics", daemon=True).start()

    def _poll(self):
        if not self.dialog.winfo_exists():
            return
        try:
            while True:
                kind, value = self.results.get_nowait()
                if kind == "diagnostics":
                    self.diagnostics_ok = value
                    if self.step == 0:
                        self.render()
                elif kind == "diagnostics_error":
                    self.diagnostics_ok = False
                    self.app._write_log("ERROR", f"Setup diagnostics failed: {value}")
                    if self.step == 0:
                        self.render()
        except queue.Empty:
            pass
        if self.step in (2, 3):
            self._refresh_camera_status()
        self.dialog.after(250, self._poll)

    def _label(self, text, *, wrap=580):
        ttk.Label(self.content, text=text, wraplength=wrap, justify="left").pack(anchor="w", pady=(0, 12))

    def render(self):
        for child in self.content.winfo_children():
            child.destroy()
        ttk.Label(self.content, text=f"{self.step + 1} / {len(self.STEPS)}  {self.STEPS[self.step]}",
                  font=("Segoe UI Semibold", 18)).pack(anchor="w", pady=(0, 15))
        self.back_button.configure(state="normal" if self.step else "disabled")
        self.next_button.configure(text="Finish setup" if self.step == 6 else "Continue")
        if self.step == 0:
            self._label("Check the Python environment, models, OSC library and phone dependencies before continuing.")
            self.status = ttk.Label(self.content, text="Diagnostics passed" if self.diagnostics_ok else "Checking… see Activity for details")
            self.status.pack(anchor="w", pady=(0, 10))
            ttk.Button(self.content, text="Run diagnostics again", command=self._run_diagnostics).pack(anchor="w")
        elif self.step == 1:
            self._label("Enable OSC in VRChat: Action Menu → OSC → Enabled. A UDP test pulse has no acknowledgement; confirm it in VRChat's OSC debug view.")
            self._entry("OSC host", self.app.osc_host)
            self._entry("OSC port", self.app.osc_port)
            ttk.Button(self.content, text="Send test pulse", command=self._test_osc).pack(anchor="w")
            self.status = ttk.Label(self.content, text="Test pulse sent" if self.osc_tested else "Send the test pulse to continue")
            self.status.pack(anchor="w", pady=(10, 0))
        elif self.step == 2:
            self._label("For a phone, start the local hub, scan its QR code on the same Wi-Fi, then tap Start camera on the phone. Allow only Private networks if Windows Firewall asks.")
            ttk.Button(self.content, text="Open phone link / QR", command=self.app._start_phone_server).pack(anchor="w", pady=(0, 8))
            ttk.Button(self.content, text="Use a local camera instead", command=self._choose_local).pack(anchor="w")
            self.status = ttk.Label(self.content, text="Waiting for a connected phone…")
            self.status.pack(anchor="w", pady=(12, 0))
            self._refresh_camera_status()
        elif self.step == 3:
            self._label("Choose the camera to use for your first tracking session. Its current connection state is shown in the main window's Activity tab.")
            self.camera_combo = ttk.Combobox(self.content, textvariable=self.camera_choice, state="readonly")
            self.camera_combo.pack(fill="x")
            self.status = ttk.Label(self.content, text="")
            self.status.pack(anchor="w", pady=(10, 0))
            self._refresh_camera_status()
        elif self.step == 4:
            self._label("Enter your standing height in meters (1.0–2.5). This sets the initial VRChat body scale; the in-session body lock follows after tracking starts.")
            self._entry("Height (m)", self.height)
        elif self.step == 5:
            self._label("Stable hip + feet is recommended while optical accuracy remains experimental.")
            ttk.Combobox(self.content, textvariable=self.trackers,
                         values=("Stable — hip + feet (recommended)", "Full — all 8 trackers (experimental)"),
                         state="readonly").pack(fill="x")
        else:
            selected = self.app.camera_sources.get(self.camera_choice.get(), "none")
            self._label(f"Camera: {selected}\nHeight: {self.height.get()} m\nTrackers: {self.trackers.get()}\n"
                        "Finish saves this profile and marks first-run setup complete. You can edit it later in Setup.")

    def _entry(self, label, variable):
        ttk.Label(self.content, text=label).pack(anchor="w")
        ttk.Entry(self.content, textvariable=variable).pack(fill="x", pady=(4, 12))

    def _test_osc(self):
        server = None
        try:
            from Lib import OSCKit
            server = OSCKit.Server(self.app.osc_host.get().strip(), int(self.app.osc_port.get()))
            server.Send(("/avatar/parameters/VRFBT_ConnectionTest", 1.0))
            server.Send(("/avatar/parameters/VRFBT_ConnectionTest", 0.0))
            self.osc_tested = True
            self.app._write_log("PASS", f"Wizard sent OSC test to {server.target[0]}:{server.target[1]}; UDP has no acknowledgement")
            self.status.configure(text="Test pulse sent; confirm it in VRChat's OSC debug view")
        except (OSError, ValueError) as exc:
            self.osc_tested = False
            messagebox.showerror("OSC test failed", str(exc), parent=self.dialog)
        finally:
            if server is not None:
                server.close()

    def _choose_local(self):
        if not self.app.discovered_local_sources:
            messagebox.showinfo("No local camera", "Refresh cameras in the main window, then try again.", parent=self.dialog)
            return
        self.use_local = True
        self._refresh_camera_status()

    def _refresh_camera_status(self):
        phones = {camera.source_id for camera in self.app.phone_hub.registry.list_cameras(False)}
        local = self.app.discovered_local_sources
        if self.step == 2:
            if self.use_local and local:
                self.status.configure(text=f"{len(local)} local camera(s) found")
            else:
                self.status.configure(text=f"{len(phones)} connected phone(s)" if phones else "Waiting for a connected phone…")
        elif self.step == 3:
            available = phones | (local if self.use_local else set())
            labels = [label for label, source in self.app.camera_sources.items() if source in available]
            self.camera_combo.configure(values=labels)
            if self.camera_choice.get() not in labels:
                self.camera_choice.set(labels[0] if labels else "")
            self.status.configure(text="Camera connected" if labels else "No available camera; go Back to connect one")

    def back(self):
        if self.step:
            self.step -= 1
            self.render()

    def next(self):
        if self.step == 0 and not self.diagnostics_ok:
            messagebox.showerror("Diagnostics required", "Resolve the failed checks in Activity, then run diagnostics again.", parent=self.dialog)
            return
        if self.step == 1 and not self.osc_tested:
            messagebox.showerror("OSC test required", "Send the test pulse before continuing.", parent=self.dialog)
            return
        if self.step == 2 and not (self.app.phone_hub.registry.list_cameras(False) or
                                   (self.use_local and self.app.discovered_local_sources)):
            messagebox.showerror("Camera required", "Connect a phone or choose an available local camera.", parent=self.dialog)
            return
        if self.step == 3:
            source = self.app.camera_sources.get(self.camera_choice.get())
            if source is None:
                messagebox.showerror("Camera required", "Select a connected camera.", parent=self.dialog)
                return
            self.app._select_source(source)
            if source.startswith("phone:"):
                self.app.manual_camera_setup.set(False)
                self.app._update_camera_setup_summary()
        if self.step == 4:
            try:
                height = float(self.height.get())
                if not 1.0 <= height <= 2.5:
                    raise ValueError
            except ValueError:
                messagebox.showerror("Invalid height", "Enter a height from 1.0 to 2.5 meters.", parent=self.dialog)
                return
            self.app.user_height.set(f"{height:.2f}")
        if self.step == 5:
            self.app.tracker_set.set(self.trackers.get())
        if self.step < 6:
            self.step += 1
            self.render()
            return
        try:
            settings, profile = self.app._configuration_from_form()
            settings = replace(settings, first_run_completed=True)
            save_configuration(settings, profile)
        except (OSError, ValueError) as exc:
            messagebox.showerror("Could not save setup", str(exc), parent=self.dialog)
            return
        self.app.loaded_settings, self.app.loaded_profile = settings, profile
        self.app._write_log("PASS", "First-run setup saved")
        self.close()
