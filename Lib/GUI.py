"""Tkinter desktop interface for VR-FBT."""

from __future__ import annotations

from datetime import datetime
import queue
import threading
import tkinter as tk
from tkinter import messagebox, ttk

from Lib.Config import CameraSetup, ConfigurationError, Profile, Settings, list_profiles, load_profile, load_settings, profile_camera_setups, profile_camera_sources, run_diagnostics, save_profile, save_settings
from Lib.Engine import EngineCallbacks, TrackingController
from Lib.RemoteCam import LocalCamera, RemoteCameraHub, discover_local_cameras, ensure_local_certificates, find_openssl


BG, PANEL, PANEL_ALT, INPUT_BG = "#0b1020", "#121a2c", "#182238", "#263653"
TEXT, MUTED, ACCENT = "#ffffff", "#bac7df", "#7c5cff"
GOOD, WARN, BAD = "#38d39f", "#f4b740", "#ff607c"
POSE_QUALITY_LABELS = {
    "Lite — fastest": "lite",
    "Full — balanced (recommended)": "full",
    "Heavy — highest accuracy": "heavy",
}
TRACKER_SET_LABELS = {
    "Stable — hip + feet (recommended)": "stable",
    "Full — all 8 trackers (experimental)": "full",
}


class VRFBTApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("VR-FBT for VRChat")
        self.geometry("1120x760")
        self.minsize(940, 640)
        self.configure(bg=BG)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        self.events: queue.Queue[tuple] = queue.Queue()
        self.loaded_settings = Settings()
        self.loaded_profile = Profile()
        self.camera_sources: dict[str, str] = {}
        self.local_cameras: list[LocalCamera] = []
        self.phone_hub = RemoteCameraHub(host='0.0.0.0', port=0, on_change=lambda: self.events.put(("phone-change",)))
        self._phone_thread = None
        self._phone_cancel = threading.Event()
        self._phone_dialog = None
        self._closing = False
        self.primary_phone_url = ""
        self.controller = TrackingController(EngineCallbacks(
            log=lambda level, msg: self.events.put(("log", level, msg)),
            state=lambda state: self.events.put(("state", state)),
            stats=lambda fps, visibility, frames: self.events.put(("stats", fps, visibility, frames)),
        ))
        self._configure_style()
        self._build_variables()
        self._build_ui()
        self._load_initial_configuration()
        self._event_timer = self.after(80, self._drain_events)

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure(".", background=BG, foreground=TEXT, fieldbackground=PANEL_ALT, borderwidth=0)
        style.configure("TFrame", background=BG)
        style.configure("Panel.TFrame", background=PANEL)
        style.configure("TLabel", background=BG, foreground=TEXT, font=("Segoe UI", 10))
        style.configure("Panel.TLabel", background=PANEL, foreground=TEXT)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("PanelMuted.TLabel", background=PANEL, foreground=MUTED)
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 23), foreground=TEXT)
        style.configure("Metric.TLabel", background=PANEL, font=("Segoe UI Semibold", 18), foreground=TEXT)
        style.configure("Accent.TButton", background=ACCENT, foreground="white", padding=(18, 10), font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton", background=[("active", "#9178ff"), ("disabled", "#3f4460")])
        style.configure("TButton", background=PANEL_ALT, foreground=TEXT, padding=(14, 9))
        style.map("TButton", background=[("active", "#24314d")])
        style.configure("TEntry", fieldbackground=INPUT_BG, foreground=TEXT, insertcolor=TEXT, padding=9, font=("Segoe UI", 11), borderwidth=1, relief="solid")
        style.map("TEntry", fieldbackground=[("disabled", PANEL_ALT), ("readonly", INPUT_BG)], foreground=[("disabled", MUTED), ("readonly", TEXT)])
        style.configure("TCombobox", fieldbackground=INPUT_BG, background=INPUT_BG, foreground=TEXT, arrowcolor=TEXT, padding=8, arrowsize=17, font=("Segoe UI", 11), borderwidth=1, relief="solid")
        style.map("TCombobox", fieldbackground=[("readonly", INPUT_BG), ("disabled", PANEL_ALT)], foreground=[("readonly", TEXT), ("disabled", MUTED)], selectbackground=[("readonly", INPUT_BG)], selectforeground=[("readonly", TEXT)], arrowcolor=[("readonly", TEXT)])
        self.option_add("*TCombobox*Listbox.background", INPUT_BG)
        self.option_add("*TCombobox*Listbox.foreground", TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", ACCENT)
        self.option_add("*TCombobox*Listbox.selectForeground", "#ffffff")
        style.configure("TCheckbutton", background=PANEL, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", PANEL)])
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=BG, foreground=MUTED, padding=(18, 10), font=("Segoe UI Semibold", 10))
        style.map("TNotebook.Tab", background=[("selected", PANEL)], foreground=[("selected", TEXT)])

    def _build_variables(self) -> None:
        self.profile_name, self.fps = tk.StringVar(value="Default"), tk.StringVar(value="30")
        self.camera_choices = [
            tk.StringVar(value="Local · Camera 0"),
            tk.StringVar(value="Off"),
            tk.StringVar(value="Off"),
        ]
        self.camera_choice = self.camera_choices[0]  # compatibility for integrations/tests
        self.phone_url = tk.StringVar(value="Phone server is stopped")
        self.manual_camera_setup = tk.BooleanVar(value=False)
        self.room_width, self.room_height, self.room_depth = tk.StringVar(value="4.0"), tk.StringVar(value="2.7"), tk.StringVar(value="4.0")
        self.camera_setup_summary = tk.StringVar(value="Automatic camera calibration · room 4.0 × 2.7 × 4.0 m")
        self.camera_setup_values: dict[str, CameraSetup] = {}
        self.show_output, self.smooth = tk.BooleanVar(value=True), tk.BooleanVar(value=True)
        self.tracking_mode, self.joint_map = tk.StringVar(value="SINGLE"), tk.StringVar(value="FULLMAP")
        self.pose_quality = tk.StringVar(value="Full — balanced (recommended)")
        self.tracker_set = tk.StringVar(value="Stable — hip + feet (recommended)")
        self.user_height = tk.StringVar(value="1.70")
        self.osc_host, self.osc_port = tk.StringVar(value="127.0.0.1"), tk.StringVar(value="9000")
        self.state, self.metric_fps, self.metric_visibility, self.metric_frames = tk.StringVar(value="READY"), tk.StringVar(value="—"), tk.StringVar(value="—"), tk.StringVar(value="0")

    def _build_ui(self) -> None:
        shell = ttk.Frame(self, padding=(26, 22)); shell.pack(fill="both", expand=True)
        header = ttk.Frame(shell); header.pack(fill="x", pady=(0, 18))
        title_group = ttk.Frame(header); title_group.pack(side="left")
        ttk.Label(title_group, text="VR-FBT", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_group, text="Low-latency VRChat full-body tracking", style="Muted.TLabel").pack(anchor="w")
        self.status_label = tk.Label(header, textvariable=self.state, bg=PANEL_ALT, fg=MUTED, font=("Segoe UI Semibold", 9), padx=14, pady=7); self.status_label.pack(side="right")
        metrics = ttk.Frame(shell); metrics.pack(fill="x", pady=(0, 16))
        self._metric(metrics, "LIVE FPS", self.metric_fps).pack(side="left", fill="x", expand=True, padx=(0, 8))
        self._metric(metrics, "POSE CONFIDENCE", self.metric_visibility).pack(side="left", fill="x", expand=True, padx=8)
        self._metric(metrics, "FRAMES SENT", self.metric_frames).pack(side="left", fill="x", expand=True, padx=(8, 0))
        notebook = ttk.Notebook(shell); notebook.pack(fill="both", expand=True)
        setup_tab, log_tab = ttk.Frame(notebook, style="Panel.TFrame", padding=22), ttk.Frame(notebook, style="Panel.TFrame", padding=16)
        notebook.add(setup_tab, text="SETUP"); notebook.add(log_tab, text="ACTIVITY & DIAGNOSTICS")
        self._build_setup(setup_tab); self._build_log(log_tab)

    def _metric(self, parent, label, variable):
        frame = ttk.Frame(parent, style="Panel.TFrame", padding=(18, 14))
        ttk.Label(frame, text=label, style="PanelMuted.TLabel", font=("Segoe UI Semibold", 8)).pack(anchor="w")
        ttk.Label(frame, textvariable=variable, style="Metric.TLabel").pack(anchor="w", pady=(3, 0))
        return frame

    def _build_setup(self, parent) -> None:
        left, right = ttk.Frame(parent, style="Panel.TFrame"), ttk.Frame(parent, style="Panel.TFrame")
        left.pack(side="left", fill="both", expand=True, padx=(0, 20)); right.pack(side="left", fill="both", expand=True)
        self.profile_combo = self._field(left, "PROFILE", self.profile_name, "combo", list_profiles())
        self.profile_combo.bind("<<ComboboxSelected>>", lambda _event: self._load_selected_profile())
        self._field(left, "TARGET FPS", self.fps)
        self.camera_combos = [
            self._field(left, "CAMERA 1 — PRIMARY", self.camera_choices[0], "combo", []),
            self._field(left, "CAMERA 2 — OPTIONAL", self.camera_choices[1], "combo", ["Off"]),
            self._field(left, "CAMERA 3 — OPTIONAL", self.camera_choices[2], "combo", ["Off"]),
        ]
        self.camera_combo = self.camera_combos[0]  # compatibility for integrations/tests
        ttk.Label(left, text="Multi-camera startup: hold a full-body T-pose for 10 seconds.", foreground=GOOD, style="Panel.TLabel").pack(anchor="w", pady=(0, 9))
        camera_actions = ttk.Frame(left, style="Panel.TFrame"); camera_actions.pack(fill="x", pady=(0, 13))
        ttk.Button(camera_actions, text="Refresh cameras", command=self._refresh_cameras).pack(side="left")
        self.phone_button = ttk.Button(camera_actions, text="Connect phone on LAN", command=self._start_phone_server); self.phone_button.pack(side="left", padx=8)
        ttk.Button(camera_actions, text="Camera layout…", command=self._show_camera_setup).pack(side="left")
        ttk.Label(left, textvariable=self.camera_setup_summary, style="PanelMuted.TLabel", wraplength=430, justify="left").pack(fill="x", pady=(0, 8))
        ttk.Label(left, textvariable=self.phone_url, style="PanelMuted.TLabel", wraplength=430, justify="left").pack(fill="x", pady=(0, 6))
        self.phone_stop_button = ttk.Button(left, text="Stop phone connection", command=self._stop_phone_server, state='disabled')
        self.phone_stop_button.pack(anchor="w", pady=(0, 13))
        self._field(right, "POSE MODEL", self.pose_quality, "combo", list(POSE_QUALITY_LABELS))
        self._field(right, "VRCHAT TRACKERS", self.tracker_set, "combo", list(TRACKER_SET_LABELS))
        self._field(right, "VRCHAT USER HEIGHT (METERS)", self.user_height)
        self._field(right, "VRCHAT OSC HOST", self.osc_host); self._field(right, "VRCHAT OSC PORT", self.osc_port)
        ttk.Label(right, text="In VRChat: Action Menu → OSC → Enabled", foreground=GOOD, style="Panel.TLabel").pack(anchor="w", pady=(0, 8))
        checks = ttk.Frame(right, style="Panel.TFrame"); checks.pack(fill="x", pady=(14, 24))
        ttk.Checkbutton(checks, text="Debug windows (3D room + annotated camera views)", variable=self.show_output).pack(anchor="w", pady=5)
        ttk.Checkbutton(checks, text="Smooth landmark motion", variable=self.smooth).pack(anchor="w", pady=5)
        actions = ttk.Frame(right, style="Panel.TFrame"); actions.pack(fill="x", pady=(10, 0))
        self.start_button = ttk.Button(actions, text="Start tracking", style="Accent.TButton", command=self._start); self.start_button.pack(side="left")
        self.stop_button = ttk.Button(actions, text="Stop", command=self._stop, state="disabled"); self.stop_button.pack(side="left", padx=8)
        self.align_button = ttk.Button(actions, text="Recalibrate & align", command=self._realign_vrchat, state="disabled"); self.align_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Save", command=self._save).pack(side="left", padx=8)
        tools = ttk.Frame(right, style="Panel.TFrame"); tools.pack(fill="x", pady=(16, 0))
        ttk.Button(tools, text="Send VRChat OSC test", command=self._test_vrchat_osc).pack(side="left")
        ttk.Button(tools, text="Run system diagnostics", command=self._diagnostics).pack(side="left", padx=8)

    def _field(self, parent, label, variable, kind="entry", values=None):
        wrapper = ttk.Frame(parent, style="Panel.TFrame"); wrapper.pack(fill="x", pady=(0, 13))
        ttk.Label(wrapper, text=label, style="PanelMuted.TLabel", font=("Segoe UI Semibold", 8)).pack(anchor="w", pady=(0, 5))
        widget = ttk.Combobox(wrapper, textvariable=variable, values=values or [], state="readonly") if kind == "combo" else ttk.Entry(wrapper, textvariable=variable)
        widget.pack(fill="x"); return widget

    def _build_log(self, parent) -> None:
        toolbar = ttk.Frame(parent, style="Panel.TFrame"); toolbar.pack(fill="x", pady=(0, 10))
        ttk.Label(toolbar, text="Runtime messages and self-check results", style="PanelMuted.TLabel").pack(side="left")
        ttk.Button(toolbar, text="Clear", command=self._clear_log).pack(side="right")
        self.log = tk.Text(parent, bg="#080d19", fg=TEXT, insertbackground=TEXT, relief="flat", padx=14, pady=12, font=("Cascadia Mono", 9), wrap="word", state="disabled"); self.log.pack(fill="both", expand=True)
        for name, color in {"INFO": "#a8b8d4", "PASS": GOOD, "WARN": WARN, "ERROR": BAD, "DEBUG": "#7887a3"}.items(): self.log.tag_configure(name, foreground=color)

    def _load_initial_configuration(self) -> None:
        try:
            settings = load_settings(); self.loaded_settings = settings; self.fps.set(str(settings.fps)); names = list_profiles()
            self.profile_name.set(next((n for n in names if n.casefold() == settings.default_profile.casefold()), names[0] if names else "Default"))
            self._load_selected_profile(); self._set_camera_options([]); self._refresh_cameras()
            self._write_log("INFO", "Configuration loaded. Run diagnostics before the first tracking session.")
        except Exception as exc:
            self._write_log("ERROR", str(exc)); messagebox.showerror("Configuration error", str(exc))

    def _load_selected_profile(self) -> None:
        try:
            p = load_profile(self.profile_name.get()); self.loaded_profile = p
            self.show_output.set(p.show_output); self.smooth.set(p.smooth)
            self.tracking_mode.set(p.tracking_mode); self.joint_map.set(p.joint_map); self.osc_host.set(p.server_ip); self.osc_port.set(str(p.server_port))
            self.user_height.set(f"{p.user_height_m:.2f}")
            self.pose_quality.set(next((label for label, value in POSE_QUALITY_LABELS.items() if value == p.pose_quality), "Full — balanced (recommended)"))
            self.tracker_set.set(next((label for label, value in TRACKER_SET_LABELS.items() if value == p.vrchat_tracker_set), "Stable — hip + feet (recommended)"))
            self.manual_camera_setup.set(p.manual_camera_setup)
            self.room_width.set(f"{p.room_size_m[0]:.2f}"); self.room_height.set(f"{p.room_size_m[1]:.2f}"); self.room_depth.set(f"{p.room_size_m[2]:.2f}")
            self.camera_setup_values = {setup.source_id: setup for setup in profile_camera_setups(p)}
            self._update_camera_setup_summary()
            self._select_sources(profile_camera_sources(p))
        except Exception as exc: self._write_log("ERROR", str(exc))

    def _configuration_from_form(self):
        sources = tuple(
            source for choice in self.camera_choices
            if choice.get() != "Off" and (source := self.camera_sources.get(choice.get()))
        )
        if not sources:
            sources = profile_camera_sources(self.loaded_profile)
        source = sources[0]
        temporary = Profile(camera_source=source, camera_sources=sources, tracking_mode="MULTI" if len(sources) > 1 else "SINGLE")
        default_setups = {setup.source_id: setup for setup in profile_camera_setups(temporary)}
        setups = tuple(self.camera_setup_values.get(item, default_setups[item]) for item in sources)
        camera_index = int(source.split(":", 1)[1]) if source.startswith("local:") else self.loaded_profile.camera_index
        settings = Settings(fps=int(self.fps.get()), default_profile=self.profile_name.get(), tcp_server=self.loaded_settings.tcp_server, udp_server=self.loaded_settings.udp_server, live_switch=self.loaded_settings.live_switch)
        profile = Profile(name=self.profile_name.get(), server_ip=self.osc_host.get().strip(), server_port=int(self.osc_port.get()), camera_index=camera_index, camera_source=source, camera_sources=sources, manual_camera_setup=self.manual_camera_setup.get(), camera_setups=setups, room_size_m=(float(self.room_width.get()), float(self.room_height.get()), float(self.room_depth.get())), show_output=self.show_output.get(), tracking_mode="MULTI" if len(sources) > 1 else "SINGLE", smooth=self.smooth.get(), pose_quality=POSE_QUALITY_LABELS.get(self.pose_quality.get(), "full"), vrchat_tracker_set=TRACKER_SET_LABELS.get(self.tracker_set.get(), "stable"), joint_map=self.joint_map.get(), joy_con_remote=self.loaded_profile.joy_con_remote, user_height_m=float(self.user_height.get()))
        return settings, profile

    def _save(self, quiet=False) -> bool:
        try:
            settings, profile = self._configuration_from_form(); save_profile(profile); save_settings(settings)
            self.profile_combo.configure(values=list_profiles()); self._write_log("PASS", f"Saved profile {profile.name} and application settings")
            if not quiet: messagebox.showinfo("Saved", "Configuration saved successfully.")
            return True
        except (ConfigurationError, ValueError) as exc:
            self._write_log("ERROR", str(exc)); messagebox.showerror("Invalid configuration", str(exc)); return False

    def _start(self) -> None:
        if not self._save(quiet=True): return
        try:
            settings, profile = self._configuration_from_form()
            self.controller.start(settings, profile, self.phone_hub); self.start_button.configure(state="disabled"); self.stop_button.configure(state="normal")
        except Exception as exc: messagebox.showerror("Could not start", str(exc))

    def _stop(self) -> None: self.controller.stop()

    def _realign_vrchat(self) -> None:
        try:
            self.controller.realign_vrchat()
            self._write_log("INFO", "VRChat recalibration requested. Stand upright, face the camera, and keep your feet visible.")
        except RuntimeError as exc:
            messagebox.showinfo("VRChat alignment", str(exc))

    def _diagnostics(self) -> None:
        self._write_log("INFO", "Running repository and runtime diagnostics…")
        ok = run_diagnostics(lambda name, passed, detail: self._write_log("PASS" if passed else "ERROR", f"{name}: {detail}"))
        self._write_log("PASS" if ok else "WARN", "Diagnostics passed" if ok else "Diagnostics completed with failures")

    def _test_vrchat_osc(self) -> None:
        server = None
        try:
            from Lib import OSCKit
            server = OSCKit.Server(self.osc_host.get().strip(), int(self.osc_port.get()))
            server.Send(("/avatar/parameters/VRFBT_ConnectionTest", 1.0))
            server.Send(("/avatar/parameters/VRFBT_ConnectionTest", 0.0))
            self._write_log("PASS", f"Sent OSC test to IPv4 {server.target[0]}:{server.target[1]}. UDP has no acknowledgement; confirm it in VRChat's OSC debug view.")
        except (OSError, ValueError) as exc:
            self._write_log("ERROR", f"VRChat OSC test failed: {exc}")
            messagebox.showerror("VRChat OSC test", str(exc))
        finally:
            if server is not None:
                server.close()

    def _drain_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "log": self._write_log(event[1], event[2])
                elif event[0] == "state": self._set_state(event[1])
                elif event[0] == "stats": self.metric_fps.set(f"{event[1]:.1f}"); self.metric_visibility.set(f"{event[2] * 100:.0f}%"); self.metric_frames.set(f"{event[3]:,}")
                elif event[0] == "camera-scan": self._set_camera_options(event[1])
                elif event[0] == "phone-change": self._set_camera_options(self.local_cameras)
                elif event[0] == 'phone-ready':
                    if not self._phone_cancel.is_set() and not self._closing:
                        self.primary_phone_url = event[1]
                        self.phone_url.set('Ready — scan the QR code to connect a phone.')
                        self.phone_button.configure(text='Show phone link', state='normal')
                        self._write_log('PASS', 'Local-network phone camera link is ready. No internet relay or phone pairing is used.')
                        self._show_phone_link()
                elif event[0] == 'phone-error':
                    self._reset_phone_ui()
                    if not self._phone_cancel.is_set():
                        self._write_log('ERROR', event[1])
                        messagebox.showerror('Phone connection', event[1])
        
        except queue.Empty: pass
        if self.primary_phone_url and not self.phone_hub.running:
            self._write_log('ERROR', 'Phone camera server stopped. Click Connect phones to start it again.')
            self._stop_phone_server()
        if not self._closing:
            self._event_timer = self.after(80, self._drain_events)

    def _set_state(self, state) -> None:
        self.state.set(state.upper()); self.status_label.configure(fg={"running": GOOD, "starting": WARN, "stopping": WARN, "error": BAD}.get(state, MUTED))
        if state == "running":
            self.align_button.configure(state="normal")
        elif state in {"stopped", "error"}:
            self.start_button.configure(state="normal"); self.stop_button.configure(state="disabled"); self.align_button.configure(state="disabled")

    def _write_log(self, level, message) -> None:
        self.log.configure(state="normal"); self.log.insert("end", f"{datetime.now():%H:%M:%S}  {level:<5}  {message.rstrip()}\n", level); self.log.see("end"); self.log.configure(state="disabled")

    def _clear_log(self) -> None:
        self.log.configure(state="normal"); self.log.delete("1.0", "end"); self.log.configure(state="disabled")

    def _refresh_cameras(self) -> None:
        self._write_log("INFO", "Looking for local cameras…")
        threading.Thread(target=lambda: self.events.put(("camera-scan", discover_local_cameras())), name="camera-discovery", daemon=True).start()

    def _set_camera_options(self, local_cameras) -> None:
        configured_sources = profile_camera_sources(self.loaded_profile)
        selected_sources = [self.camera_sources.get(choice.get()) for choice in self.camera_choices]
        if not any(selected_sources):
            selected_sources = list(configured_sources)
        cameras = list(local_cameras)
        for configured in configured_sources:
            if not any(camera.source_id == configured for camera in cameras) and configured.startswith("local:"):
                index = int(configured.split(":", 1)[1])
                cameras.append(LocalCamera(index, f"Camera {index}"))
        self.local_cameras = cameras
        sources = {camera.display_name: camera.source_id for camera in cameras}
        for camera in self.phone_hub.registry.list_cameras():
            label = camera.display_name
            if label in sources:
                label = f'{label} [{camera.device_id[:8]}]'
            sources[label] = camera.source_id
        for configured in configured_sources:
            if configured not in sources.values():
                kind, identifier = configured.split(":", 1)
                sources[f"Saved · {kind} {identifier[:24]} (offline)"] = configured
        self.camera_sources = sources
        values = list(sources)
        self.camera_combos[0].configure(values=values)
        for combo in self.camera_combos[1:]:
            combo.configure(values=["Off", *values])
        self._select_sources(tuple(source for source in selected_sources if source))
        if sources and self.camera_choice.get() not in sources:
            self.camera_choice.set(next(iter(sources)))
        local_count = len(cameras); phone_count = len(self.phone_hub.registry.list_cameras(False))
        self._write_log("PASS", f"Camera list updated: {local_count} local, {phone_count} connected phone(s)")

    def _select_source(self, source_id: str) -> None:
        self._select_sources((source_id,))

    def _select_sources(self, source_ids) -> None:
        for index, choice in enumerate(self.camera_choices):
            source_id = source_ids[index] if index < len(source_ids) else None
            display = next((name for name, value in self.camera_sources.items() if value == source_id), None)
            choice.set(display if display else (choice.get() if index == 0 else "Off"))

    def _update_camera_setup_summary(self) -> None:
        mode = "Manual fixed cameras" if self.manual_camera_setup.get() else "Automatic camera calibration"
        self.camera_setup_summary.set(
            f"{mode} · room {self.room_width.get()} × {self.room_height.get()} × {self.room_depth.get()} m"
        )

    def _show_camera_setup(self) -> None:
        sources = tuple(
            source for choice in self.camera_choices
            if choice.get() != "Off" and (source := self.camera_sources.get(choice.get()))
        ) or profile_camera_sources(self.loaded_profile)
        defaults = {
            setup.source_id: setup
            for setup in profile_camera_setups(Profile(
                camera_source=sources[0],
                camera_sources=sources,
                tracking_mode="MULTI" if len(sources) > 1 else "SINGLE",
            ))
        }
        dialog = tk.Toplevel(self)
        dialog.title("Camera and room layout")
        dialog.configure(bg=BG)
        dialog.transient(self)
        dialog.grab_set()
        box = ttk.Frame(dialog, padding=22); box.pack(fill="both", expand=True)
        ttk.Label(box, text="Tracking room and fixed camera layout", font=("Segoe UI Semibold", 16)).grid(row=0, column=0, columnspan=9, sticky="w", pady=(0, 10))
        manual = tk.BooleanVar(value=self.manual_camera_setup.get())
        ttk.Checkbutton(box, text="Use these manual camera transforms (all selected cameras)", variable=manual).grid(row=1, column=0, columnspan=9, sticky="w", pady=(0, 10))

        room_values = [tk.StringVar(value=value.get()) for value in (self.room_width, self.room_height, self.room_depth)]
        ttk.Label(box, text="ROOM W / H / D (m)", style="Muted.TLabel").grid(row=2, column=0, sticky="w")
        for column, variable in enumerate(room_values, start=1):
            ttk.Entry(box, textvariable=variable, width=8).grid(row=2, column=column, padx=4, sticky="ew")

        headings = ("CAMERA", "X", "Y", "Z", "YAW", "PITCH", "ROLL", "H-FOV")
        for column, heading in enumerate(headings):
            ttk.Label(box, text=heading, style="Muted.TLabel", font=("Segoe UI Semibold", 8)).grid(row=3, column=column, padx=4, pady=(16, 5), sticky="w")
        setup_variables = {}
        for row, source in enumerate(sources, start=4):
            setup = self.camera_setup_values.get(source, defaults[source])
            values = [tk.StringVar(value=f"{value:.3f}") for value in (*setup.position, *setup.rotation, setup.horizontal_fov)]
            setup_variables[source] = values
            ttk.Label(box, text=f"CAM {row - 3} · {source.split(':', 1)[-1][:18]}").grid(row=row, column=0, padx=4, pady=5, sticky="w")
            for column, variable in enumerate(values, start=1):
                ttk.Entry(box, textvariable=variable, width=9).grid(row=row, column=column, padx=4, pady=5, sticky="ew")

        ttk.Label(
            box,
            text="Coordinates use the room center as X=0, Z=0 and the floor as Y=0.\nYaw 0° looks toward +Z; yaw -90° looks toward -X. Pitch tilts vertically. FOV is the camera's horizontal field of view.",
            style="Muted.TLabel",
            justify="left",
        ).grid(row=8, column=0, columnspan=9, sticky="w", pady=(14, 10))

        def apply_layout():
            try:
                room = tuple(float(value.get()) for value in room_values)
                if not (1.0 <= room[0] <= 20.0 and 1.8 <= room[1] <= 6.0 and 1.0 <= room[2] <= 20.0):
                    raise ValueError("Room must be 1–20 m wide/deep and 1.8–6 m high")
                setups = {}
                for source, variables in setup_variables.items():
                    numbers = [float(variable.get()) for variable in variables]
                    if not 25.0 <= numbers[6] <= 120.0:
                        raise ValueError("Horizontal FOV must be between 25° and 120°")
                    setups[source] = CameraSetup(source, tuple(numbers[:3]), tuple(numbers[3:6]), numbers[6])
            except ValueError as exc:
                messagebox.showerror("Invalid camera layout", str(exc), parent=dialog)
                return
            self.manual_camera_setup.set(manual.get())
            self.room_width.set(f"{room[0]:.2f}"); self.room_height.set(f"{room[1]:.2f}"); self.room_depth.set(f"{room[2]:.2f}")
            self.camera_setup_values.update(setups)
            self._update_camera_setup_summary()
            dialog.destroy()

        actions = ttk.Frame(box); actions.grid(row=9, column=0, columnspan=9, sticky="e", pady=(8, 0))
        ttk.Button(actions, text="Cancel", command=dialog.destroy).pack(side="left", padx=6)
        ttk.Button(actions, text="Apply layout", style="Accent.TButton", command=apply_layout).pack(side="left")

    def _start_phone_server(self) -> None:
        if self.primary_phone_url and self.phone_hub.running:
            self._show_phone_link()
            return
        if self._phone_thread and self._phone_thread.is_alive():
            return
        self._phone_cancel.clear()
        self.phone_hub = RemoteCameraHub(host='0.0.0.0', port=0, on_change=lambda: self.events.put(('phone-change',)))
        self.phone_button.configure(text='Connecting…', state='disabled')
        self.phone_stop_button.configure(state='normal')
        self.phone_url.set('Creating a secure local Wi-Fi phone link…')
        self._write_log('INFO', 'Opening the phone camera service only on this PC and its local network.')
        self._phone_thread = threading.Thread(target=self._open_phone_link, daemon=True, name='phone-link-startup')
        self._phone_thread.start()

    def _open_phone_link(self):
        try:
            openssl = find_openssl()
            self.events.put(('log', 'INFO', f'OpenSSL is generating/checking the local HTTPS certificate: {openssl}'))
            certificate, private_key, _ca_certificate = ensure_local_certificates()
            self.phone_hub.start(certificate, private_key)
            if self._phone_cancel.is_set():
                return
            camera_urls = self.phone_hub.local_urls()
            if not camera_urls:
                raise RuntimeError('No local-network address is available for the phone camera service.')
            camera_url = camera_urls[0].replace('/?token=', '/#token=')
            self.events.put(('phone-ready', camera_url))
        except Exception as exc:
            self.phone_hub.stop()
            self.events.put(('phone-error', str(exc)))
        finally:
            if self._phone_cancel.is_set():
                self.phone_hub.stop()

    def _show_phone_link(self):
        if self._phone_dialog and self._phone_dialog.winfo_exists():
            self._phone_dialog.lift()
            return
        dialog = self._phone_dialog = tk.Toplevel(self)
        dialog.title('Connect your phone')
        dialog.configure(bg=BG)
        dialog.transient(self)
        box = ttk.Frame(dialog, padding=24)
        box.pack(fill='both', expand=True)
        ttk.Label(box, text='Connect on local Wi-Fi', font=('Segoe UI Semibold', 16)).pack(pady=(0, 12))
        self._add_qr(box, dialog, self.primary_phone_url, box_size=5)
        ttk.Label(
            box,
            text='Scan the QR code, continue past the local certificate warning, then press Start camera.\nAllow camera access and select the phone under Camera Source.',
            justify='center',
        ).pack(pady=12)
        link = ttk.Entry(box, width=65)
        link.insert(0, self.primary_phone_url)
        link.configure(state='readonly')
        link.pack(fill='x')
        ttk.Button(box, text='Copy local phone link', command=self._copy_phone_url).pack(pady=10)
        ttk.Label(
            box,
            text='Phone and PC must use the same local network. No ADB, pairing, cloud service,\nor internet video relay is used. Allow only Private networks if Windows Firewall asks.',
            style='Muted.TLabel',
            justify='center',
        ).pack()

    @staticmethod
    def _add_qr(parent, dialog, value: str, box_size: int = 4):
        try:
            import qrcode
            from PIL import ImageTk
            qr = qrcode.QRCode(box_size=box_size, border=4)
            qr.add_data(value)
            qr.make(fit=True)
            image = ImageTk.PhotoImage(qr.make_image(fill_color='black', back_color='white').convert('RGB'))
            if not hasattr(dialog, 'qr_images'):
                dialog.qr_images = []
            dialog.qr_images.append(image)
            dialog.qr_image = image
            ttk.Label(parent, image=image).pack()
        except ImportError:
            ttk.Label(parent, text='Copy the link below and open it on your phone.').pack()

    def _reset_phone_ui(self):
        self.primary_phone_url = ''
        self.phone_url.set('Phone connection is stopped')
        self.phone_button.configure(text='Connect phone on LAN', state='normal')
        self.phone_stop_button.configure(state='disabled')
        if self._phone_dialog and self._phone_dialog.winfo_exists():
            self._phone_dialog.destroy()
        self._phone_dialog = None

    def _stop_phone_server(self):
        self._phone_cancel.set()
        if self.controller.running:
            self.controller.stop()
        self.phone_hub.stop()
        self._reset_phone_ui()

    def _copy_phone_url(self) -> None:
        value = self.primary_phone_url
        if value.startswith(("http://", "https://")):
            self.clipboard_clear(); self.clipboard_append(value); self._write_log("PASS", "Phone URL copied")
        else:
            messagebox.showinfo("Phone server", "Start the phone server first.")

    def _on_close(self) -> None:
        self._closing = True
        self.after_cancel(self._event_timer)
        self._stop_phone_server()
        if self.controller.running: self.controller.stop(); self.controller.wait(2.0)
        if self._phone_thread:
            self._phone_thread.join(timeout=6)
        self.destroy()


def launch() -> None:
    VRFBTApp().mainloop()
