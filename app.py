"""Live RTSP person counter with editable polygon zones (Windows desktop MVP)."""

from __future__ import annotations

import json
import csv
import re
import shutil
import threading
import time
from serial import Serial, SerialException
from datetime import datetime
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk
from ultralytics import YOLO


DEFAULT_URL = "rtsp://192.168.137.231:8554/live/0"
MODEL_FILE = "yolo11n.pt"
ZONES_FILE = Path("zones.json")
PEOPLE_DIR = Path("people")
ATTENDANCE_FILE = Path("attendance.csv")
LIGHTING_CONFIG_FILE = Path("lighting_config.json")
FACE_DETECTOR_MODEL = Path("models/face_detection_yunet_2023mar.onnx")
FACE_RECOGNITION_MODEL = Path("models/face_recognition_sface_2021dec.onnx")
ATTENDANCE_INTERVAL_SECONDS = 2
FACE_SIMILARITY_THRESHOLD = 0.363
PERSON_CLASS_ID = 0
LIGHTING_UPDATE_INTERVAL_SECONDS = 0.25
ESP32_ALLOWED_GPIO_PINS = {18, 19, 21, 22, 23, 25, 26, 27, 32, 33}


class LightingController:
    """Small serial client for the ESP32's ``SET <gpio> <percent>`` protocol."""

    def __init__(self) -> None:
        self.connection: Serial | None = None

    @property
    def connected(self) -> bool:
        return self.connection is not None and self.connection.is_open

    def connect(self, port: str) -> None:
        self.disconnect()
        self.connection = Serial(port=port, baudrate=115200, timeout=0.2, write_timeout=0.2)
        # Opening a USB serial port resets many ESP32 DevKit boards.
        time.sleep(2.0)
        self.connection.reset_input_buffer()
        self.connection.write(b"PING\n")
        self.connection.flush()
        response = self.connection.readline().decode("ascii", errors="replace").strip()
        if response != "PONG":
            self.disconnect()
            raise SerialException(f"No ESP32 response on {port}. Check the firmware and close Arduino Serial Monitor.")

    def set_brightness(self, gpio_pin: int, percent: int) -> None:
        if not self.connected:
            return
        self.connection.write(f"SET {gpio_pin} {max(0, min(100, int(percent)))}\n".encode("ascii"))

    def disconnect(self) -> None:
        if self.connection is not None:
            try:
                self.connection.close()
            finally:
                self.connection = None


class VideoSource:
    """Reads the newest frame on a worker thread so the UI stays responsive."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.capture: cv2.VideoCapture | None = None
        self.frame: np.ndarray | None = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.stop()
        self.stop_event.clear()
        self.error = None
        self.capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        if not self.capture.isOpened():
            self.error = "Could not open the feed. Check the URL, Wi-Fi, and that the phone is streaming."
            self.capture.release()
            self.capture = None
            return
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self) -> None:
        assert self.capture is not None
        while not self.stop_event.is_set():
            ok, frame = self.capture.read()
            if ok:
                with self.lock:
                    self.frame = frame
            else:
                self.error = "Lost the camera stream. Trying to reconnect…"
                time.sleep(0.2)

    def latest(self) -> np.ndarray | None:
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1)
        if self.capture is not None:
            self.capture.release()
        self.thread = None
        self.capture = None


class ZoneCounterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Hall Occupancy — Camera & Zones")
        self.root.geometry("1280x820")
        self.root.minsize(900, 620)

        self.source: VideoSource | None = None
        self.model: YOLO | None = None
        self.model_loading = False
        self.detection_enabled = tk.BooleanVar(value=True)
        self.attendance_enabled = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value="Enter a camera URL and press Connect.")
        self.detector_status = tk.StringVar(value="YOLO: not connected")
        self.zone_counts: dict[str, int] = {}
        self.zone_person_names: dict[str, set[str]] = {}
        self.zones: list[dict] = []
        self.pending_points: list[tuple[int, int]] = []
        self.latest_frame: np.ndarray | None = None
        self.display_scale = 1.0
        self.display_offset = (0, 0)
        self.photo: ImageTk.PhotoImage | None = None
        self.last_inference = 0.0
        self.latest_boxes: list[tuple[int, int, int, int, float]] = []
        self.latest_face_labels: list[tuple[int, int, str]] = []
        self.face_detector = None
        self.face_recognizer = None
        self.person_embeddings: dict[str, list[np.ndarray]] = {}
        self.face_model_loading = False
        self.last_attendance_sample = 0.0
        self.attendance_records: dict[tuple[str, str], dict[str, str | int]] = {}
        self.lighting_config = self._load_lighting_config()
        self.lighting_controller = LightingController()
        self.lighting_status = tk.StringVar(value="ESP32: disconnected")
        self.last_lighting_update = 0.0
        self.lighting_levels: dict[str, float] = {}
        self.lighting_targets: dict[str, int] = {}
        self.zone_last_occupied: dict[str, float] = {}
        self._load_attendance_records()
        PEOPLE_DIR.mkdir(exist_ok=True)
        self._build_ui()
        self._load_zones()
        self.root.protocol("WM_DELETE_WINDOW", self._close)
        self.root.after(50, self._tick)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)

        controls = ttk.Frame(outer)
        controls.pack(fill="x", pady=(0, 8))
        ttk.Label(controls, text="Camera feed (RTSP or webcam index):").pack(side="left")
        self.url_entry = ttk.Entry(controls, width=58)
        self.url_entry.insert(0, DEFAULT_URL)
        self.url_entry.pack(side="left", padx=8)
        ttk.Button(controls, text="Connect", command=self._connect).pack(side="left", padx=3)
        ttk.Button(controls, text="Disconnect", command=self._disconnect).pack(side="left", padx=3)

        toolbar = ttk.Frame(outer)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Button(toolbar, text="New zone", command=self._new_zone).pack(side="left", padx=(0, 5))
        ttk.Button(toolbar, text="Finish polygon (Enter)", command=self._finish_polygon).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Undo point", command=self._undo_point).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Save zones", command=self._save_zones).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Clear zones", command=self._clear_zones).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Lighting & events", command=self._open_lighting_config).pack(side="left", padx=5)
        ttk.Button(toolbar, text="Connect ESP32", command=self._connect_lighting_controller).pack(side="left", padx=5)
        ttk.Label(toolbar, textvariable=self.lighting_status).pack(side="left", padx=5)
        ttk.Checkbutton(toolbar, text="2-second face attendance", variable=self.attendance_enabled, command=self._toggle_attendance).pack(side="right", padx=(8, 0))
        ttk.Checkbutton(toolbar, text="YOLO person detection", variable=self.detection_enabled).pack(side="right")

        enrollment = ttk.LabelFrame(outer, text="Register a person", padding=(8, 5))
        enrollment.pack(fill="x", pady=(0, 8))
        ttk.Label(enrollment, text="Name:").pack(side="left")
        self.person_name_entry = ttk.Entry(enrollment, width=28)
        self.person_name_entry.pack(side="left", padx=7)
        ttk.Button(enrollment, text="Select photos and add", command=self._register_person_photos).pack(side="left", padx=5)
        ttk.Label(enrollment, text="Select multiple JPG/PNG photos; images are copied and numbered in people/<name>/.", foreground="#555").pack(side="left", padx=8)

        body = ttk.Frame(outer)
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, background="#15191e", highlightthickness=0, cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Button-1>", self._on_canvas_click)
        self.canvas.bind("<Button-3>", lambda _event: self._finish_polygon())
        self.zone_panel = ttk.LabelFrame(body, text="Zone occupancy", padding=10, width=220)
        self.zone_panel.pack(side="right", fill="y", padx=(10, 0))
        self.zone_panel.pack_propagate(False)
        self.count_labels: dict[str, ttk.Label] = {}

        ttk.Label(outer, textvariable=self.detector_status, anchor="w").pack(fill="x", pady=(8, 0))
        ttk.Label(outer, textvariable=self.status, anchor="w").pack(fill="x")
        ttk.Label(outer, text="Click vertices over the camera image. Press Enter or right-click to close a polygon. Zone counts use the detected person's foot point.", foreground="#555").pack(fill="x")
        self.root.bind("<Return>", lambda _event: self._finish_polygon())
        self.root.bind("<Control-s>", lambda _event: self._save_zones())

    def _load_lighting_config(self) -> dict:
        config = {
            "room_name": "Demo Room",
            "serial_port": "COM12",
            "prelight_seconds": 10,
            "standby_brightness": 30,
            "occupied_brightness": 100,
            "vacancy_delay_seconds": 5,
            "fade_seconds": 3,
            "event_end_behavior": "off",
            "outputs": {},
            "events": [],
        }
        if not LIGHTING_CONFIG_FILE.exists():
            return config
        try:
            saved = json.loads(LIGHTING_CONFIG_FILE.read_text(encoding="utf-8"))
            if isinstance(saved, dict):
                config.update(saved)
        except (OSError, json.JSONDecodeError) as exc:
            self.status.set(f"Could not read lighting configuration: {exc}")
        config["outputs"] = config.get("outputs") or {}
        config["events"] = config.get("events") or []
        return config

    def _ensure_zone_outputs(self) -> None:
        outputs = self.lighting_config.setdefault("outputs", {})
        for zone in self.zones:
            zone_name = str(zone["name"])
            output_name = re.sub(r"[^\w-]+", "_", zone_name, flags=re.UNICODE).strip("_") or zone_name
            outputs.setdefault(zone_name, {"bulb_id": f"B_{output_name}", "gpio_pin": ""})

    def _open_lighting_config(self) -> None:
        if getattr(self, "lighting_window", None) is not None and self.lighting_window.winfo_exists():
            self.lighting_window.lift()
            return

        self._ensure_zone_outputs()
        window = tk.Toplevel(self.root)
        self.lighting_window = window
        window.title("Lighting & Event Configuration")
        window.geometry("900x650")
        window.minsize(760, 540)
        outer = ttk.Frame(window, padding=12)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Connect over USB serial. During scheduled events unoccupied zones stay at standby; occupied zones fade to the occupied level.", foreground="#40566f", wraplength=850).pack(anchor="w", pady=(0, 8))

        notebook = ttk.Notebook(outer)
        notebook.pack(fill="both", expand=True)
        behavior_tab = ttk.Frame(notebook, padding=14)
        outputs_tab = ttk.Frame(notebook, padding=14)
        events_tab = ttk.Frame(notebook, padding=14)
        notebook.add(behavior_tab, text="Room behavior")
        notebook.add(outputs_tab, text="Zone outputs")
        notebook.add(events_tab, text="Event schedule")

        behavior_fields = [
            ("Room name", "room_name", "Demo Room"),
            ("ESP32 serial port", "serial_port", "COM12"),
            ("Pre-light default (seconds)", "prelight_seconds", "10"),
            ("Standby brightness (%)", "standby_brightness", "30"),
            ("Occupied-zone brightness (%)", "occupied_brightness", "100"),
            ("Vacancy delay (seconds)", "vacancy_delay_seconds", "5"),
            ("Fade duration (seconds)", "fade_seconds", "3"),
        ]
        self.lighting_entries: dict[str, ttk.Entry] = {}
        for row, (label_text, key, fallback) in enumerate(behavior_fields):
            ttk.Label(behavior_tab, text=label_text).grid(row=row, column=0, sticky="w", padx=(0, 16), pady=7)
            entry = ttk.Entry(behavior_tab, width=28)
            entry.insert(0, str(self.lighting_config.get(key, fallback)))
            entry.grid(row=row, column=1, sticky="w", pady=7)
            self.lighting_entries[key] = entry

        ttk.Label(behavior_tab, text="After the event ends:").grid(row=len(behavior_fields), column=0, sticky="w", padx=(0, 16), pady=7)
        self.event_end_var = tk.StringVar(value=self.lighting_config.get("event_end_behavior", "off"))
        ttk.Combobox(behavior_tab, textvariable=self.event_end_var, state="readonly", width=26,
                     values=("off", "standby")).grid(row=len(behavior_fields), column=1, sticky="w", pady=7)
        ttk.Label(behavior_tab, text="off = fade to 0% at event end; standby = keep the room at its standby level.", foreground="#555").grid(
            row=len(behavior_fields) + 1, column=0, columnspan=2, sticky="w", pady=(12, 4)
        )

        ttk.Label(outputs_tab, text="Map each camera zone to one LED. Enter the ESP32 GPIO pin wired to that zone's LED; supported pins are 18, 19, 21, 22, 23, 25, 26, 27, 32, and 33.", wraplength=810).pack(anchor="w", pady=(0, 10))
        output_columns = ("zone", "bulb", "gpio")
        self.output_tree = ttk.Treeview(outputs_tab, columns=output_columns, show="headings", height=9, selectmode="browse")
        for column, heading, width in (("zone", "Zone", 210), ("bulb", "Bulb ID", 210), ("gpio", "GPIO pin", 140)):
            self.output_tree.heading(column, text=heading)
            self.output_tree.column(column, width=width, anchor="w")
        self.output_tree.pack(fill="x", expand=False)
        output_form = ttk.Frame(outputs_tab)
        output_form.pack(fill="x", pady=10)
        ttk.Label(output_form, text="Zone:").pack(side="left")
        self.output_zone_var = tk.StringVar()
        self.output_zone_combo = ttk.Combobox(output_form, textvariable=self.output_zone_var, state="readonly", width=16)
        self.output_zone_combo.pack(side="left", padx=(5, 12))
        ttk.Label(output_form, text="Bulb ID:").pack(side="left")
        self.output_bulb_entry = ttk.Entry(output_form, width=16)
        self.output_bulb_entry.pack(side="left", padx=5)
        ttk.Label(output_form, text="GPIO:").pack(side="left", padx=(8, 0))
        self.output_gpio_entry = ttk.Entry(output_form, width=10)
        self.output_gpio_entry.pack(side="left", padx=5)
        ttk.Button(output_form, text="Apply mapping", command=self._apply_output_mapping).pack(side="left", padx=5)
        self.output_zone_combo.bind("<<ComboboxSelected>>", self._load_selected_output)
        self.output_tree.bind("<<TreeviewSelect>>", self._load_selected_output_row)
        self._refresh_output_tree()

        ttk.Label(events_tab, text="Add multiple dated events. Times use your laptop's local clock (24-hour format).", wraplength=810).pack(anchor="w", pady=(0, 10))
        event_columns = ("name", "start", "end", "prelight")
        self.event_tree = ttk.Treeview(events_tab, columns=event_columns, show="headings", height=8, selectmode="browse")
        for column, heading, width in (("name", "Event", 190), ("start", "Starts", 190), ("end", "Ends", 190), ("prelight", "Pre-light (s)", 110)):
            self.event_tree.heading(column, text=heading)
            self.event_tree.column(column, width=width, anchor="w")
        self.event_tree.pack(fill="both", expand=True)
        event_form = ttk.LabelFrame(events_tab, text="New event", padding=8)
        event_form.pack(fill="x", pady=(10, 0))
        self.event_name_entry = self._event_entry(event_form, "Name", 0, 20)
        self.event_start_entry = self._event_entry(event_form, "Start (YYYY-MM-DD HH:MM)", 1, 19)
        self.event_end_entry = self._event_entry(event_form, "End (YYYY-MM-DD HH:MM)", 2, 19)
        self.event_prelight_entry = self._event_entry(event_form, "Pre-light seconds", 3, 7)
        self.event_prelight_entry.insert(0, str(self.lighting_config.get("prelight_seconds", 10)))
        event_buttons = ttk.Frame(event_form)
        event_buttons.grid(row=4, column=0, columnspan=4, sticky="w", pady=(8, 0))
        ttk.Button(event_buttons, text="Add event", command=self._add_lighting_event).pack(side="left")
        ttk.Button(event_buttons, text="Remove selected", command=self._remove_lighting_event).pack(side="left", padx=8)
        self._refresh_event_tree()

        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))
        self.lighting_save_status = tk.StringVar(value="Save mappings and events, then connect the ESP32.")
        ttk.Label(footer, textvariable=self.lighting_save_status).pack(side="left", fill="x", expand=True)
        ttk.Button(footer, text="Save settings", command=self._save_lighting_config).pack(side="right")

        def close_window() -> None:
            self.lighting_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", close_window)

    @staticmethod
    def _event_entry(parent: ttk.Widget, label: str, column: int, width: int) -> ttk.Entry:
        cell = ttk.Frame(parent)
        cell.grid(row=0, column=column, sticky="w", padx=(0, 8))
        ttk.Label(cell, text=label).pack(anchor="w")
        entry = ttk.Entry(cell, width=width)
        entry.pack(anchor="w", pady=(3, 0))
        return entry

    def _refresh_output_tree(self) -> None:
        if not hasattr(self, "output_tree"):
            return
        self.output_tree.delete(*self.output_tree.get_children())
        zone_names = [str(zone["name"]) for zone in self.zones]
        self.output_zone_combo["values"] = zone_names
        for index, zone_name in enumerate(zone_names):
            mapping = self.lighting_config["outputs"].get(zone_name, {})
            self.output_tree.insert("", "end", iid=str(index), values=(zone_name, mapping.get("bulb_id", ""), mapping.get("gpio_pin", "")))
        if zone_names and not self.output_zone_var.get():
            self.output_zone_var.set(zone_names[0])
            self._load_selected_output()

    def _load_selected_output_row(self, _event=None) -> None:
        selection = self.output_tree.selection()
        if selection:
            values = self.output_tree.item(selection[0], "values")
            if values:
                self.output_zone_var.set(values[0])
                self._load_selected_output()

    def _load_selected_output(self, _event=None) -> None:
        zone_name = self.output_zone_var.get()
        mapping = self.lighting_config["outputs"].get(zone_name, {})
        self.output_bulb_entry.delete(0, "end")
        self.output_bulb_entry.insert(0, mapping.get("bulb_id", ""))
        self.output_gpio_entry.delete(0, "end")
        self.output_gpio_entry.insert(0, str(mapping.get("gpio_pin", "")))

    def _apply_output_mapping(self) -> None:
        zone_name = self.output_zone_var.get()
        bulb_id = self.output_bulb_entry.get().strip()
        gpio_pin = self.output_gpio_entry.get().strip()
        if not zone_name:
            messagebox.showerror("Zone output", "Create and save at least one zone before mapping outputs.", parent=self.lighting_window)
            return
        if not bulb_id:
            messagebox.showerror("Bulb ID", "Enter an output ID such as B_1.", parent=self.lighting_window)
            return
        if gpio_pin and not gpio_pin.isdigit():
            messagebox.showerror("GPIO pin", "Enter a numeric GPIO pin.", parent=self.lighting_window)
            return
        if gpio_pin and int(gpio_pin) not in ESP32_ALLOWED_GPIO_PINS:
            messagebox.showerror("GPIO pin", "Use one of these ESP32 output pins: 18, 19, 21, 22, 23, 25, 26, 27, 32, or 33.", parent=self.lighting_window)
            return
        if gpio_pin and any(
            other_zone != zone_name and str(other_mapping.get("gpio_pin", "")) == gpio_pin
            for other_zone, other_mapping in self.lighting_config["outputs"].items()
        ):
            messagebox.showerror("GPIO pin", "Each zone must use a different GPIO pin.", parent=self.lighting_window)
            return
        self.lighting_config["outputs"][zone_name] = {"bulb_id": bulb_id, "gpio_pin": gpio_pin}
        self._refresh_output_tree()
        self.output_zone_var.set(zone_name)
        self._load_selected_output()
        self.lighting_save_status.set("Mapping updated in this window. Select Save settings to write it to disk.")

    def _refresh_event_tree(self) -> None:
        if not hasattr(self, "event_tree"):
            return
        self.event_tree.delete(*self.event_tree.get_children())
        events = sorted(self.lighting_config.get("events", []), key=lambda event: event.get("start", ""))
        self.lighting_config["events"] = events
        for index, event in enumerate(events):
            self.event_tree.insert("", "end", iid=str(index), values=(event["name"], event["start"].replace("T", " "), event["end"].replace("T", " "), event["prelight_seconds"]))

    def _add_lighting_event(self) -> None:
        name = self.event_name_entry.get().strip()
        try:
            start = datetime.strptime(self.event_start_entry.get().strip(), "%Y-%m-%d %H:%M")
            end = datetime.strptime(self.event_end_entry.get().strip(), "%Y-%m-%d %H:%M")
            prelight = int(self.event_prelight_entry.get().strip())
        except ValueError:
            messagebox.showerror("Event details", "Use a name, dates in YYYY-MM-DD HH:MM format, and a whole number of pre-light seconds.", parent=self.lighting_window)
            return
        if not name or end <= start or prelight < 0:
            messagebox.showerror("Event details", "Enter a name, an end time after the start time, and a non-negative pre-light delay.", parent=self.lighting_window)
            return
        self.lighting_config.setdefault("events", []).append({
            "name": name,
            "start": start.isoformat(timespec="minutes"),
            "end": end.isoformat(timespec="minutes"),
            "prelight_seconds": prelight,
        })
        self._refresh_event_tree()
        self.event_name_entry.delete(0, "end")
        self.event_start_entry.delete(0, "end")
        self.event_end_entry.delete(0, "end")
        self.lighting_save_status.set("Event added in this window. Select Save settings to write it to disk.")

    def _remove_lighting_event(self) -> None:
        selection = self.event_tree.selection()
        if not selection:
            return
        index = int(selection[0])
        events = self.lighting_config.get("events", [])
        if 0 <= index < len(events):
            events.pop(index)
            self._refresh_event_tree()
            self.lighting_save_status.set("Event removed in this window. Select Save settings to write it to disk.")

    def _save_lighting_config(self) -> None:
        try:
            room_name = self.lighting_entries["room_name"].get().strip()
            serial_port = self.lighting_entries["serial_port"].get().strip()
            prelight = int(self.lighting_entries["prelight_seconds"].get().strip())
            standby = int(self.lighting_entries["standby_brightness"].get().strip())
            occupied = int(self.lighting_entries["occupied_brightness"].get().strip())
            vacancy = int(self.lighting_entries["vacancy_delay_seconds"].get().strip())
            fade = float(self.lighting_entries["fade_seconds"].get().strip())
        except ValueError:
            messagebox.showerror("Lighting settings", "Enter whole numbers for delays/brightness and a number for fade duration.", parent=self.lighting_window)
            return
        if not room_name or not serial_port or prelight < 0 or not (0 <= standby <= occupied <= 100) or vacancy < 0 or fade <= 0:
            messagebox.showerror("Lighting settings", "Use a room name, non-negative delays, 0–100 brightness with standby no higher than occupied, and a positive fade duration.", parent=self.lighting_window)
            return
        self.lighting_config.update({
            "room_name": room_name,
            "serial_port": serial_port,
            "prelight_seconds": prelight,
            "standby_brightness": standby,
            "occupied_brightness": occupied,
            "vacancy_delay_seconds": vacancy,
            "fade_seconds": fade,
            "event_end_behavior": self.event_end_var.get(),
        })
        self.lighting_targets.clear()
        try:
            LIGHTING_CONFIG_FILE.write_text(json.dumps(self.lighting_config, indent=2), encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Save lighting settings", str(exc), parent=self.lighting_window)
            return
        self.lighting_save_status.set(f"Saved to {LIGHTING_CONFIG_FILE.resolve()}.")
        self.status.set("Lighting configuration saved.")

    def _connect_lighting_controller(self) -> None:
        port = str(self.lighting_config.get("serial_port", "COM12")).strip()
        try:
            self.lighting_controller.connect(port)
        except (SerialException, OSError, ValueError) as exc:
            self.lighting_status.set(f"ESP32: connection failed ({exc})")
            self.status.set("ESP32 connection failed. Close Arduino Serial Monitor and check the COM port.")
            return
        self.lighting_status.set(f"ESP32: connected on {port}")
        self.lighting_targets.clear()
        self.status.set(f"ESP32 connected on {port}. Save zone GPIO mappings in Lighting & events.")

    def _update_lighting(self) -> None:
        now = time.monotonic()
        if now - self.last_lighting_update < LIGHTING_UPDATE_INTERVAL_SECONDS:
            return
        self.last_lighting_update = now

        # Pre-light all mapped zones before a scheduled event, and keep standby
        # levels through the event. Without a schedule, lights are demand-driven.
        wall_now = datetime.now()
        in_event_window = False
        event_finished_today = False
        for event in self.lighting_config.get("events", []):
            try:
                start = datetime.fromisoformat(event["start"])
                end = datetime.fromisoformat(event["end"])
                prelight = int(event.get("prelight_seconds", self.lighting_config.get("prelight_seconds", 10)))
                if start.timestamp() - prelight <= wall_now.timestamp() <= end.timestamp():
                    in_event_window = True
                if start.date() == wall_now.date() and end < wall_now:
                    event_finished_today = True
            except (KeyError, TypeError, ValueError):
                continue

        standby = int(self.lighting_config.get("standby_brightness", 30))
        occupied = int(self.lighting_config.get("occupied_brightness", 100))
        vacancy_delay = float(self.lighting_config.get("vacancy_delay_seconds", 5))
        fade_seconds = max(0.1, float(self.lighting_config.get("fade_seconds", 3)))
        output_map = self.lighting_config.get("outputs", {})
        for zone in self.zones:
            zone_name = str(zone["name"])
            mapping = output_map.get(zone_name, {})
            try:
                gpio_pin = int(mapping.get("gpio_pin", ""))
            except (TypeError, ValueError):
                continue
            if self.zone_counts.get(zone_name, 0) > 0:
                self.zone_last_occupied[zone_name] = now
                target = occupied
            elif now - self.zone_last_occupied.get(zone_name, -1e9) < vacancy_delay:
                target = occupied
            else:
                keep_standby = in_event_window or (
                    event_finished_today and self.lighting_config.get("event_end_behavior") == "standby"
                )
                target = standby if keep_standby else 0

            current = self.lighting_levels.get(zone_name, 0.0)
            step = max(1.0, 100.0 * LIGHTING_UPDATE_INTERVAL_SECONDS / fade_seconds)
            if current < target:
                current = min(float(target), current + step)
            elif current > target:
                current = max(float(target), current - step)
            self.lighting_levels[zone_name] = current
            rounded = int(round(current))
            if self.lighting_targets.get(zone_name) != rounded:
                if self.lighting_controller.connected:
                    try:
                        self.lighting_controller.set_brightness(gpio_pin, rounded)
                    except (SerialException, OSError) as exc:
                        self.lighting_status.set(f"ESP32: serial error ({exc})")
                        self.lighting_controller.disconnect()
                self.lighting_targets[zone_name] = rounded

    def _connect(self) -> None:
        self._disconnect()
        url = self.url_entry.get().strip()
        if not url:
            messagebox.showerror("Camera URL", "Enter an RTSP URL or a webcam index such as 0.")
            return
        source = VideoSource(int(url) if url.isdigit() else url)
        source.start()
        if source.error:
            self.status.set(source.error)
            messagebox.showerror("Camera connection", source.error)
            return
        self.source = source
        self.status.set(f"Connected to {url}. Loading the lightweight YOLO model on CPU…")
        self.detector_status.set("YOLO: loading yolo11n.pt on CPU…")
        self._load_model_async()

    def _load_model_async(self) -> None:
        if self.model is not None or self.model_loading:
            return
        self.model_loading = True

        def load() -> None:
            try:
                model = YOLO(MODEL_FILE)
                self.root.after(0, lambda: self._model_ready(model, None))
            except Exception as exc:  # surfaced in UI; common first-run cause is unavailable model download
                self.root.after(0, lambda error=str(exc): self._model_ready(None, error))

        threading.Thread(target=load, daemon=True).start()

    def _model_ready(self, model: YOLO | None, error: str | None) -> None:
        self.model_loading = False
        if model is not None:
            self.model = model
            self.detector_status.set("YOLO ready — waiting for the next camera frame")
            self.status.set("Draw seating zones, or enable face attendance after adding enrollment photos to people/<name>/.")
        else:
            self.detector_status.set(f"YOLO failed to load: {error}")
            self.status.set("Check internet access for the first model download, then reconnect.")

    def _toggle_attendance(self) -> None:
        if not self.attendance_enabled.get():
            self.zone_person_names = {}
            self.latest_face_labels = []
            return
        if self.face_recognizer is not None:
            self.status.set("Face attendance is active. Each enrolled person is counted at most once every 2 seconds.")
            return
        if self.face_model_loading:
            return
        self._start_face_model_training()

    def _start_face_model_training(self) -> None:
        if self.face_model_loading:
            self.status.set("Face model is already refreshing; please wait for it to finish.")
            return
        self.face_model_loading = True
        self.face_recognizer = None
        self.face_detector = None
        self.person_embeddings = {}
        self.status.set("Loading enrollment photos from people/<name>/…")
        threading.Thread(target=self._train_face_model, daemon=True).start()

    def _train_face_model(self) -> None:
        try:
            if not FACE_DETECTOR_MODEL.exists() or not FACE_RECOGNITION_MODEL.exists():
                raise FileNotFoundError("YuNet/SFace model files are missing from the project's models folder.")
            detector = cv2.FaceDetectorYN.create(str(FACE_DETECTOR_MODEL), "", (320, 320), 0.65, 0.3, 5000)
            recognizer = cv2.FaceRecognizerSF.create(str(FACE_RECOGNITION_MODEL), "")
            embeddings: dict[str, list[np.ndarray]] = {}
            for person_dir in sorted(path for path in PEOPLE_DIR.iterdir() if path.is_dir()):
                for image_path in sorted(person_dir.iterdir()):
                    if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                        continue
                    image = cv2.imread(str(image_path))
                    if image is None:
                        continue
                    face = self._largest_face_row(detector, image)
                    if face is None:
                        continue
                    embedding = self._face_embedding(recognizer, image, face)
                    if embedding is not None:
                        embeddings.setdefault(person_dir.name, []).append(embedding)
            if not embeddings:
                raise ValueError("No usable faces found. Add clear, front-facing JPG/PNG photos under people/<Person Name>/ and retry.")
            self.root.after(0, lambda: self._face_model_ready(detector, recognizer, embeddings, None))
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._face_model_ready(None, None, {}, error))

    @staticmethod
    def _largest_face_row(detector, image: np.ndarray) -> np.ndarray | None:
        height, width = image.shape[:2]
        detector.setInputSize((width, height))
        _status, faces = detector.detect(image)
        if faces is None or len(faces) == 0:
            return None
        return max(faces, key=lambda face: float(face[2] * face[3]))

    @staticmethod
    def _face_embedding(recognizer, image: np.ndarray, face: np.ndarray) -> np.ndarray | None:
        aligned = recognizer.alignCrop(image, face)
        feature = recognizer.feature(aligned)
        feature = np.asarray(feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(feature))
        if norm == 0:
            return None
        return feature / norm

    def _face_model_ready(self, detector, recognizer, embeddings: dict[str, list[np.ndarray]], error: str | None) -> None:
        self.face_model_loading = False
        if error:
            self.attendance_enabled.set(False)
            self.status.set(f"Face attendance not ready: {error}")
            return
        self.face_detector = detector
        self.face_recognizer = recognizer
        self.person_embeddings = embeddings
        self.status.set(f"Face attendance ready for {len(embeddings)} enrolled person(s); sampling every 2 seconds.")

    def _load_attendance_records(self) -> None:
        if not ATTENDANCE_FILE.exists():
            try:
                with ATTENDANCE_FILE.open("w", newline="", encoding="utf-8") as file:
                    csv.DictWriter(file, fieldnames=["date", "person", "times_seen", "first_seen", "last_seen", "zones"]).writeheader()
            except OSError:
                pass
            return
        try:
            with ATTENDANCE_FILE.open("r", newline="", encoding="utf-8-sig") as file:
                for row in csv.DictReader(file):
                    key = (row["date"], row["person"])
                    self.attendance_records[key] = {
                        "date": row["date"], "person": row["person"],
                        "times_seen": int(row["times_seen"]),
                        "first_seen": row["first_seen"], "last_seen": row["last_seen"],
                        "zones": row.get("zones", ""),
                    }
        except (OSError, ValueError, KeyError):
            self.status.set(f"Could not read {ATTENDANCE_FILE}; a new attendance file will be created when attendance is recorded.")

    def _write_attendance(self, person_zones: dict[str, set[str]]) -> None:
        now = datetime.now()
        date_text = now.date().isoformat()
        time_text = now.strftime("%H:%M:%S")
        for person, zones in person_zones.items():
            key = (date_text, person)
            zone_text = "; ".join(sorted(zones)) if zones else "Unassigned"
            record = self.attendance_records.get(key)
            if record is None:
                self.attendance_records[key] = {
                    "date": date_text, "person": person, "times_seen": 1,
                    "first_seen": time_text, "last_seen": time_text,
                    "zones": zone_text,
                }
            else:
                record["times_seen"] = int(record["times_seen"]) + 1
                record["last_seen"] = time_text
                previous_zones = set(str(record.get("zones", "")).split("; "))
                record["zones"] = "; ".join(sorted((previous_zones | zones) - {"", "Unassigned"})) or zone_text
        with ATTENDANCE_FILE.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=["date", "person", "times_seen", "first_seen", "last_seen", "zones"])
            writer.writeheader()
            for key in sorted(self.attendance_records):
                writer.writerow(self.attendance_records[key])

    def _sample_attendance(self, frame: np.ndarray) -> None:
        if not self.attendance_enabled.get() or self.face_recognizer is None:
            return
        now = time.monotonic()
        if now - self.last_attendance_sample < ATTENDANCE_INTERVAL_SECONDS:
            return
        self.last_attendance_sample = now
        height, width = frame.shape[:2]
        self.face_detector.setInputSize((width, height))
        _status, faces = self.face_detector.detect(frame)
        people_by_zone: dict[str, set[str]] = {zone["name"]: set() for zone in self.zones}
        unassigned: set[str] = set()
        self.latest_face_labels = []
        if faces is not None:
            for face in faces:
                x, y, width, height = np.rint(face[:4]).astype(int)
                embedding = self._face_embedding(self.face_recognizer, frame, face)
                if embedding is None:
                    continue
                matches: list[tuple[str, float]] = []
                for person_name, references in self.person_embeddings.items():
                    similarity = max(
                        float(self.face_recognizer.match(embedding.reshape(1, -1), reference.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))
                        for reference in references
                    )
                    matches.append((person_name, similarity))
                person, similarity = max(matches, key=lambda item: item[1], default=(None, -1.0))
                if person and similarity >= FACE_SIMILARITY_THRESHOLD:
                    x, y, width, height = int(x), int(y), int(width), int(height)
                    x, y = max(0, x), max(0, y)
                    face_center = (x + width // 2, y + height // 2)
                    face_bottom = (x + width // 2, y + height)
                    associated_box = next((box for box in self.latest_boxes if box[0] <= face_center[0] <= box[2] and box[1] <= face_center[1] <= box[3]), None)
                    matched_zone = self._zone_for_person_box(associated_box) if associated_box else self._zone_for_points((face_bottom, face_center))
                    if matched_zone:
                        people_by_zone.setdefault(matched_zone, set()).add(person)
                    else:
                        unassigned.add(person)
                    self.latest_face_labels.append((x, y, person))
        self.zone_person_names = {name: names for name, names in people_by_zone.items() if names}
        for person in unassigned:
            people_by_zone.setdefault("Unassigned", set()).add(person)
        if unassigned:
            self.zone_person_names["Unassigned"] = unassigned
        person_zones: dict[str, set[str]] = {}
        for zone_name, names in people_by_zone.items():
            for name in names:
                person_zones.setdefault(name, set()).add(zone_name)
        if person_zones:
            self._write_attendance(person_zones)
        identified_names = sorted(person_zones)
        self.status.set(f"Attendance sample {datetime.now().strftime('%H:%M:%S')}: {', '.join(identified_names) if identified_names else 'no enrolled faces recognized'}.")

    def _zone_for_points(self, points: tuple[tuple[int, int], ...]) -> str | None:
        for point in points:
            for zone in self.zones:
                polygon = np.asarray(zone["points"], dtype=np.int32)
                if cv2.pointPolygonTest(polygon, point, False) >= 0:
                    return zone["name"]
        return None

    def _zone_for_person_box(self, box: tuple[int, int, int, int, float] | None) -> str | None:
        if box is None:
            return None
        x1, y1, x2, y2 = box[:4]
        center_x = (x1 + x2) // 2
        height = max(1, y2 - y1)
        # The detector box can extend a few pixels below the seating polygon;
        # use progressively higher foot points to tolerate that boundary noise.
        candidates = tuple((center_x, int(y2 - height * inset)) for inset in (0.0, 0.06, 0.12, 0.2))
        return self._zone_for_points(candidates)

    def _disconnect(self) -> None:
        if self.source:
            self.source.stop()
            self.source = None

    def _new_zone(self) -> None:
        if self.pending_points:
            messagebox.showinfo("Finish current polygon", "Finish or undo the polygon already in progress first.")
            return
        name = simpledialog.askstring("New seating zone", "Zone name:", parent=self.root)
        if name and name.strip():
            self.pending_name = name.strip()
            self.status.set(f"Drawing {self.pending_name}: click at least three polygon vertices.")

    def _register_person_photos(self) -> None:
        raw_name = self.person_name_entry.get().strip()
        person_name = re.sub(r"[^\w.-]+", "-", raw_name, flags=re.UNICODE).strip("._-")
        if not person_name:
            messagebox.showerror("Person name", "Enter a valid person name first.")
            self.person_name_entry.focus_set()
            return
        selected_files = filedialog.askopenfilenames(
            parent=self.root,
            title=f"Select enrollment photos for {person_name}",
            filetypes=[("Face photos", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")],
        )
        if not selected_files:
            return
        person_dir = PEOPLE_DIR / person_name
        person_dir.mkdir(parents=True, exist_ok=True)
        pattern = re.compile(rf"^{re.escape(person_name)}_(\d+)$", re.IGNORECASE)
        existing_numbers = [int(match.group(1)) for file in person_dir.iterdir() if (match := pattern.match(file.stem))]
        next_number = max(existing_numbers, default=0) + 1
        copied = 0
        skipped = 0
        for selected in selected_files:
            source = Path(selected)
            if source.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp"}:
                skipped += 1
                continue
            destination = person_dir / f"{person_name}_{next_number:03d}{source.suffix.lower()}"
            try:
                shutil.copy2(source, destination)
                next_number += 1
                copied += 1
            except OSError as exc:
                messagebox.showerror("Copy enrollment photo", f"Could not copy {source.name}: {exc}")
                return
        if copied == 0:
            messagebox.showerror("Enrollment photos", "No supported image files were selected.")
            return
        summary = f"Copied {copied} photo(s) to {person_dir}."
        if skipped:
            summary += f" Skipped {skipped} unsupported file(s)."
        self.status.set(summary + " Refreshing face model…")
        self._start_face_model_training()

    def _on_canvas_click(self, event: tk.Event) -> None:
        if not getattr(self, "pending_name", None):
            self.status.set("Press New zone before drawing a polygon.")
            return
        if self.latest_frame is None:
            self.status.set("Connect a camera before drawing zones.")
            return
        x = int((event.x - self.display_offset[0]) / self.display_scale)
        y = int((event.y - self.display_offset[1]) / self.display_scale)
        height, width = self.latest_frame.shape[:2]
        if 0 <= x < width and 0 <= y < height:
            self.pending_points.append((x, y))

    def _finish_polygon(self) -> None:
        if len(self.pending_points) < 3:
            if self.pending_points:
                self.status.set("A zone needs at least three points.")
            return
        name = getattr(self, "pending_name", None) or f"Zone {len(self.zones) + 1}"
        self.zones.append({"name": name, "points": self.pending_points.copy()})
        self.pending_points.clear()
        self.pending_name = None
        self._refresh_zone_list()
        self.status.set(f"Added {name}. Save zones when you are done.")

    def _undo_point(self) -> None:
        if self.pending_points:
            self.pending_points.pop()

    def _clear_zones(self) -> None:
        if self.zones and not messagebox.askyesno("Clear zones", "Remove all zones from this session?"):
            return
        self.zones.clear()
        self.pending_points.clear()
        self.pending_name = None
        self._refresh_zone_list()

    def _save_zones(self) -> None:
        try:
            ZONES_FILE.write_text(json.dumps({"zones": self.zones}, indent=2), encoding="utf-8")
            self.status.set(f"Saved {len(self.zones)} zone(s) to {ZONES_FILE.resolve()}.")
        except OSError as exc:
            messagebox.showerror("Save zones", str(exc))

    def _load_zones(self) -> None:
        if not ZONES_FILE.exists():
            return
        try:
            data = json.loads(ZONES_FILE.read_text(encoding="utf-8"))
            self.zones = data.get("zones", [])
            self._refresh_zone_list()
        except (OSError, json.JSONDecodeError) as exc:
            self.status.set(f"Could not load {ZONES_FILE}: {exc}")

    def _refresh_zone_list(self) -> None:
        for child in self.zone_panel.winfo_children():
            child.destroy()
        self.count_labels = {}
        self.zone_name_labels: dict[str, ttk.Label] = {}
        if not self.zones:
            ttk.Label(self.zone_panel, text="No zones yet\n\nClick New zone to start.", justify="left").pack(anchor="w")
        for zone in self.zones:
            frame = ttk.Frame(self.zone_panel)
            frame.pack(fill="x", pady=5)
            ttk.Label(frame, text=zone["name"], font=("Segoe UI", 10, "bold")).pack(anchor="w")
            label = ttk.Label(frame, text="0 people", font=("Segoe UI", 16))
            label.pack(anchor="w")
            self.count_labels[zone["name"]] = label
            names_label = ttk.Label(frame, text="Identified: —", wraplength=190, justify="left", foreground="#40566f")
            names_label.pack(anchor="w", pady=(2, 0))
            self.zone_name_labels[zone["name"]] = names_label
        ttk.Separator(self.zone_panel, orient="horizontal").pack(fill="x", pady=(8, 6))
        ttk.Label(self.zone_panel, text="Outside zones", font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.unassigned_name_label = ttk.Label(self.zone_panel, text="Identified: —", wraplength=190, justify="left", foreground="#40566f")
        self.unassigned_name_label.pack(anchor="w", pady=(2, 0))

    def _detect(self, frame: np.ndarray) -> None:
        self._sample_attendance(frame)
        if not self.detection_enabled.get():
            self.latest_boxes = []
            self.zone_counts = {zone["name"]: 0 for zone in self.zones}
            self.detector_status.set("YOLO: paused")
            return
        if self.model is None or self.model_loading:
            return
        now = time.monotonic()
        if now - self.last_inference < 0.35:
            return
        self.last_inference = now
        # Downsize input for practical CPU inference on a laptop and later Raspberry Pi evaluation.
        h, w = frame.shape[:2]
        scale = min(1.0, 960 / max(h, w))
        infer_frame = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        results = self.model.predict(infer_frame, classes=[PERSON_CLASS_ID], conf=0.20, imgsz=960, device="cpu", verbose=False)
        boxes: list[tuple[int, int, int, int, float]] = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = (box.xyxy[0].cpu().numpy() / scale).astype(int)
            boxes.append((int(x1), int(y1), int(x2), int(y2), float(box.conf[0])))
        self.latest_boxes = boxes
        self.detector_status.set(f"YOLO ready — {len(boxes)} person(s) detected in full view")
        counts = {zone["name"]: 0 for zone in self.zones}
        for box in boxes:
            matched_zone = self._zone_for_person_box(box)
            if matched_zone:
                counts[matched_zone] += 1
        self.zone_counts = counts

    def _render(self, frame: np.ndarray) -> None:
        image = frame.copy()
        for zone_index, zone in enumerate(self.zones):
            points = np.array(zone["points"], dtype=np.int32)
            color = ((70 + 61 * zone_index) % 255, (210 + 47 * zone_index) % 255, (100 + 83 * zone_index) % 255)
            cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
            anchor = tuple(points[0])
            cv2.putText(image, f"{zone['name']}: {self.zone_counts.get(zone['name'], 0)}", anchor, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
        if self.pending_points:
            pts = np.array(self.pending_points, dtype=np.int32)
            for point in self.pending_points:
                cv2.circle(image, point, 5, (0, 230, 255), -1)
            if len(pts) > 1:
                cv2.polylines(image, [pts], False, (0, 230, 255), 2, cv2.LINE_AA)
        for x1, y1, x2, y2, confidence in self.latest_boxes:
            cv2.rectangle(image, (x1, y1), (x2, y2), (40, 220, 70), 2)
            cv2.putText(image, f"person {confidence:.2f}", (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40, 220, 70), 2)
        for x, y, name in self.latest_face_labels:
            cv2.putText(image, name, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)

        canvas_w = max(100, self.canvas.winfo_width())
        canvas_h = max(100, self.canvas.winfo_height())
        h, w = image.shape[:2]
        self.display_scale = min(canvas_w / w, canvas_h / h)
        resized = cv2.resize(image, (int(w * self.display_scale), int(h * self.display_scale)))
        self.display_offset = ((canvas_w - resized.shape[1]) // 2, (canvas_h - resized.shape[0]) // 2)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(Image.fromarray(rgb))
        self.canvas.delete("all")
        self.canvas.create_image(self.display_offset[0], self.display_offset[1], image=self.photo, anchor="nw")
        for name, label in self.count_labels.items():
            label.configure(text=f"{self.zone_counts.get(name, 0)} people")
        for name, label in getattr(self, "zone_name_labels", {}).items():
            detected = sorted(self.zone_person_names.get(name, set()))
            label.configure(text=f"Identified: {', '.join(detected) if detected else '—'}")
        if hasattr(self, "unassigned_name_label"):
            unassigned = sorted(self.zone_person_names.get("Unassigned", set()))
            self.unassigned_name_label.configure(text=f"Identified: {', '.join(unassigned) if unassigned else '—'}")

    def _tick(self) -> None:
        if self.source:
            frame = self.source.latest()
            if frame is not None:
                self.latest_frame = frame
                try:
                    self._detect(frame)
                except Exception as exc:
                    self.detector_status.set(f"Detection error: {exc}")
                    self.latest_boxes = []
                self._render(frame)
                self._update_lighting()
            elif self.source.error:
                self.status.set(self.source.error)
        self.root.after(50, self._tick)

    def _close(self) -> None:
        self._disconnect()
        if self.lighting_controller.connected:
            for mapping in self.lighting_config.get("outputs", {}).values():
                try:
                    pin = int(mapping.get("gpio_pin", ""))
                    if pin in ESP32_ALLOWED_GPIO_PINS:
                        self.lighting_controller.set_brightness(pin, 0)
                except (TypeError, ValueError, SerialException, OSError):
                    pass
        self.lighting_controller.disconnect()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    ZoneCounterApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
